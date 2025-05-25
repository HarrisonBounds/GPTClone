from gpt2_clone import GPT
import tiktoken
import torch
import json
import time

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


#Get Device
if torch.cuda.is_available():
        device = "cuda"
else:
    device = "cpu"

print("Device: ", device)

#Load hyperparameters
with open('gpt_config.json', 'r') as f:
        hyperparameters = json.load(f)


train_loader = DataLoaderTest(B=16, T=128)

torch.set_float32_matmul_precision("high")

#Get logits
model = GPT(hyperparameters)
model.to(device)
model = torch.compile(model)

#Optimizer - GPT3 hyperparamters
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, )

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000

    tps = (train_loader.B * train_loader.T) / (t1-t0)

    print(f"Step: {i+1}, Loss: {loss.item()}, time: {dt} ms, tok/sec: {tps}")
