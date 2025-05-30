import os
import json
import time
import math
import numpy as np
import torch
import tiktoken # Assuming this is used for tokenization, though not explicitly in the provided code.
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from gpt2_clone import GPT


def load_tokens(filepath):
    npt = np.load(filepath)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split, data_folder="edu_fineweb10B"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_folder = data_folder

        assert split in {'train', 'val'}, "Split must be 'train' or 'val'."

        self.shards = self._get_shard_filepaths(split)
        self.current_shard_idx = 0
        self.tokens = self._load_current_shard()
        self.current_position = self.B * self.T * self.process_rank

    def _get_shard_filepaths(self, split):
        shards = os.listdir(self.data_folder)
        shards = [s for s in shards if split in s]
        shards.sort()
        shards = [os.path.join(self.data_folder, s) for s in shards]
        return shards

    def _load_current_shard(self):
        return load_tokens(self.shards[self.current_shard_idx])

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes

        # Wrap around to the next shard if the current position exceeds token length
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.tokens = self._load_current_shard()
            self.current_position = self.B * self.T * self.process_rank

        return x, y


def get_learning_rate(it, config, min_lr):
    warmup_steps = config["warmup_steps"]
    max_steps = config["max_steps"]
    max_lr = config["max_lr"]

    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, "Decay ratio out of bounds."

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# --- Distributed Training Setup ---

def setup_distributed_training():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA is required for DDP."
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device

def load_hyperparameters(config_path='gpt_config.json'):
    with open(config_path, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters

def set_seeds(seed=1449):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_model_and_optimizer(hyperparameters, device, ddp, ddp_local_rank):
    model = GPT(hyperparameters)
    model.to(device)
    model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters["max_lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )
    return model, optimizer

# --- Main Training Loop ---

def main():
    # Setup distributed training and device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_distributed_training()
    print(f"Using device: {device}")

    # Load hyperparameters and set seeds
    hyperparameters = load_hyperparameters()
    set_seeds()

    # Calculate gradient accumulation steps
    total_batch_size = hyperparameters["total_batch_size"]
    B = hyperparameters["batch_size"]
    T = hyperparameters["seq_len"]
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"I am GPU rank {ddp_rank}")

    # Initialize data loader
    train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

    # Set up model and optimizer
    torch.set_float32_matmul_precision("high")
    model, optimizer = create_model_and_optimizer(hyperparameters, device, ddp, ddp_local_rank)
    min_lr = hyperparameters["max_lr"] * 0.1

    print("Starting training loop...")
    for step in range(hyperparameters["max_steps"]):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            # Scale the loss and accumulate
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            if ddp:
                # Synchronize gradients only on the last micro-step
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            loss.backward()

        if ddp:
            # Average loss across all GPUs
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update learning rate
        lr = get_learning_rate(step, hyperparameters, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Calculate tokens per second
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt

        if master_process:
            print(f"Step: {step+1}, Loss: {loss_accum.item():.4f}, LR: {lr:.2e}, Time: {dt * 1000:.2f} ms, Tokens/sec: {tokens_per_sec:.2f}")

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()