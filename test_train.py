from gpt2_clone import GPT
import tiktoken
import torch
import json
import time
import math
import os
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#------------------------------------------------------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderTest:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        #Get shard names
        data_folder = "edu_fineweb10B"
        shards = os.listdir(data_folder)

        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_folder, s) for s in shards]
        self.shards = shards

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank 


    def next_batch(self):
        B, T = self.B, self.T

        #Leave on the CPU for now
        buf = self.tokens[self.current_position :  self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes #advancement based on gpu

        #Wrap back arounf if next batch is out of bounds
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y
#------------------------------------------------------------------------------------------------------------

#Manual LR scheduler
def get_lr(it, config, min_lr):
    
    if it < config["warmup_steps"]:
        return config["max_lr"] * (it+1) / config["warmup_steps"]
    
    if it > config["max_steps"]:
        return min_lr
    
    decay_ratio = (it - config["warmup_steps"]) / (config["max_steps"] - config["warmup_steps"])

    assert 0 <= decay_ratio <= 1

    coeff =  0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (config["max_lr"] - min_lr)
    
#------------------------------------------------------------------------------------------------------------

#Distributed Data Parallel (DDP), using multiple GPUs for training
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
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

print("Device: ", device)

#Load hyperparameters
with open('gpt_config.json', 'r') as f:
        hyperparameters = json.load(f)

#Set seed for reproducibility and multiple GPU use
torch.manual_seed(1449)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1449)

total_batch_size = 524288
B = 64
T = 1024
#gradient accumulation
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

#Only print if GPU is master process
if master_process:
    print(f"grad accumulation: {grad_accum_steps}")

print(f"I am GPU rank {ddp_rank}")

train_loader = DataLoaderTest(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

torch.set_float32_matmul_precision("high")

#Create model
model = GPT(hyperparameters)
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

min_lr = hyperparameters["max_lr"] * 0.1

#Optimizer - GPT3 hyperparamters
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["max_lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

print("starting training loop...")
for step in range(hyperparameters["max_steps"]):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps #scale the loss down and apply mean
        loss_accum += loss.detach() #values instead of tensors
        if ddp:
            #Toggle bool to not sync until the last step to reduce overhead
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        loss.backward()

    if ddp:
        #Create avg of loss across GPUs
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) 

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient norm clipping

    lr = get_lr(step, hyperparameters, min_lr)
    for param_group in optimizer.param_groups:
         param_group["lr"] = lr

    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0

    tps = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt

    if master_process:
        print(f"Step: {step+1}, Loss: {loss_accum.item()}, LR: {lr}, time: {dt * 1000} ms, tok/sec: {tps}")

if ddp:
    destroy_process_group()