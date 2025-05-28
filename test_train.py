from gpt2_clone import GPT
import tiktoken
import torch
import json
import time
import math
import os
from torch.distributed import init_process_group, destroy_process_group

#------------------------------------------------------------------------------------------------------------
class DataLoaderTest:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        #Get data
        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = 0

        print("Number of tokens loaded: ", len(self.tokens))
        print(f"1 epochs = {len(self.tokens) // (B * T)} batches")


    def next_batch(self):
        B, T = self.B, self.T

        #Leave on the CPU for now
        buf = self.tokens[self.current_position :  self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T

        #Wrap back arounf if next batch is out of bounds
        if self.current_position + (B * T + 1) > len(self.tokens):
             self.current_position = 0

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

total_batch_size = 524288
B = 32
T = 1024
#gradient accumulation
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

#Only print if GPU is master process
if master_process:
    print(f"grad accumulation: {grad_accum_steps}")

print(f"I am GPU rank {ddp_rank}")

import sys ; sys.exit(0)

train_loader = DataLoaderTest(B=B, T=T)

torch.set_float32_matmul_precision("high")

#Get logits
model = GPT(hyperparameters)
model.to(device)
model = torch.compile(model)

min_lr = hyperparameters["max_lr"] * 0.1

#Optimizer - GPT3 hyperparamters
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["max_lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

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
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient norm clipping

    lr = get_lr(step, hyperparameters, min_lr)
    for param_group in optimizer.param_groups:
         param_group["lr"] = lr

    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0

    tps = (train_loader.B * train_loader.T * grad_accum_steps) / dt

    print(f"Step: {step+1}, Loss: {loss.item()}, LR: {lr}, time: {dt * 1000} ms, tok/sec: {tps}")
