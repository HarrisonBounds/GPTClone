import os
import json
import time
import math
import numpy as np
import torch
import tiktoken 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from gpt2_clone import GPT
from hellaswag import render_example, iterate_examples, get_most_likely_row

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

        self.reset()

    def _get_shard_filepaths(self, split):
        shards = os.listdir(self.data_folder)
        shards = [s for s in shards if split in s]
        shards.sort()
        shards = [os.path.join(self.data_folder, s) for s in shards]
        return shards

    def _load_current_shard(self):
        return load_tokens(self.shards[self.current_shard_idx])

    def reset(self):
        self.current_shard_idx = 0
        self.tokens = self._load_current_shard()
        self.current_position = self.B * self.T * self.process_rank

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


def sample_from_model(model, device, enc, num_samples=5, max_new_tokens=100, temperature=0.9, top_k=None):
    model.eval() # Set the model to evaluation mode
    start_ids = enc.encode("Hello, a language model is")
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    sample_outputs = []
    for _ in range(num_samples):
        # Sample from the model
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode the generated tokens
        sample = enc.decode(y[0].tolist())
        sample_outputs.append(sample)
    
    model.train() # Set the model back to training mode
    return sample_outputs

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, ddp, hyperparameters):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict() if ddp else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': hyperparameters,
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    # Setup distributed training and device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_distributed_training()
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    log_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    writer = None
    if master_process:
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be written to: {log_dir}")

    # Model checkpoints path
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if master_process and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    samples_dir = os.path.join(log_dir, "samples")
    sample_file = "all_samples.txt"
    if master_process and not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    # Load hyperparameters and set seeds
    hyperparameters = load_hyperparameters()
    enc = tiktoken.get_encoding("gpt2")
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
    val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val") 

    # Set up model and optimizer
    torch.set_float32_matmul_precision("high")
    model, optimizer = create_model_and_optimizer(hyperparameters, device, ddp, ddp_local_rank)
    min_lr = hyperparameters["max_lr"] * 0.1

    print("Starting training loop...")
    last_step = False
    global_step = 0
    for epoch in range(hyperparameters["num_epochs"]):
        last_step = False

        for step in range(hyperparameters["max_steps"]):
            t0 = time.time()

            if step == hyperparameters["max_steps"] - 1:
                last_step = True

            if step % 5000 == 0 or last_step:
                if master_process:
                    save_checkpoint(model, optimizer, epoch, global_step + 1, checkpoint_dir, ddp, hyperparameters)

            if step % 500 == 0 or last_step:
                #Validation
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 5

                    for i in range(val_loss_steps):
                        #print(f"val step {i}")
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)

                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            logits, loss = model(x, y)

                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()

                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

                if master_process:
                    print(f"Validation Loss:{val_loss_accum.item():.4f}")
                    writer.add_scalar("Loss/val", val_loss_accum.item(), global_step)
            
            if step % 500 == 0 or last_step:
                model.eval()
                num_return_sequences = 3
                max_new_tokens = 32
                tokens = enc.encode("Hello, a language model is")
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                xgen = tokens.to(device)
                sample_rng = torch.Generator(device=device)
                sample_rng.manual_seed(42 + ddp_rank)

                while xgen.size(1) < max_new_tokens:
                    with torch.no_grad():
                        logits, loss = model(xgen)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                        xcol = torch.gather(topk_indices, -1, ix)
                        xgen = torch.cat((xgen, xcol), dim=1)

                if master_process:
                    with open(f"{samples_dir}/{sample_file}", 'w') as f:
                        f.write(f"\n=== Step {step} ===\n")

                        for i in range(num_return_sequences):
                            tokens = xgen[i, :max_new_tokens].tolist()
                            decoded = enc.decode(tokens)
                            #print(f"Rank: {ddp_rank}\nSample: {decoded}")

                            f.write(f"{decoded}\n")

            if step % 500 == 0 or last_step:
                num_correct_norm = 0
                num_total = 0
                for i, example in enumerate(iterate_examples("val")):
                    if i % ddp_world_size != ddp_rank:
                        continue
                    # render the example into tokens and labels
                    _, tokens, mask, label = render_example(example)
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    # get the logits
                    with torch.no_grad():
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            logits, loss = model(tokens)
                        pred_norm = get_most_likely_row(tokens, mask, logits)
                    num_total += 1
                    num_correct_norm += int(pred_norm == label)
                # reduce the stats across all processes
                if ddp:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                    num_total = num_total.item()
                    num_correct_norm = num_correct_norm.item()
                acc_norm = num_correct_norm / num_total
                if master_process:
                    print(f"HellaSwag accuracy: {acc_norm:.4f}")
                    writer.add_scalar("hellaswag/acc", acc_norm, global_step)
                    

            optimizer.zero_grad()
            loss_accum = 0.0
            model.train()

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
            lr = get_learning_rate(global_step, hyperparameters, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0

            # Calculate tokens per second
            tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt

            if master_process:
                print(f"Epoch {epoch+1}, Step: {step+1}, Loss: {loss_accum.item():.4f}, LR: {lr:.2e}, Time: {dt * 1000:.2f} ms, Tokens/sec: {tokens_per_sec:.2f}")

                # Log metrics to TensorBoard
                #Training
                writer.add_scalar("Loss/train", loss_accum.item(), global_step)
                writer.add_scalar("Tokens_per_second", tokens_per_sec, global_step)
                writer.add_scalar("Learning_rate", lr, global_step)
                writer.add_scalar("Time_per_step_ms", dt * 1000, global_step)

            global_step += 1

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()