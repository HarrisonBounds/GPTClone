from gpt2_clone import GPT
import tiktoken
import torch
import json


#------------------------------------------------------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")

#Get Device
if torch.cuda.is_available():
        device = "cuda"
else:
    device = "cpu"

print("Device: ", device)

#Load hyperparameters
with open('gpt_config.json', 'r') as f:
        hyperparameters = json.load(f)

#Get data
with open('input.txt', 'r') as f:
      text = f.read()

text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

#Get logits
model = GPT(hyperparameters)
model.to(device)

#Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step{i+1} : Loss = {loss.item()}")




