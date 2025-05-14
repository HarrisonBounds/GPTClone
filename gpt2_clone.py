import json
import torch.nn as nn


#Processes each token in the sequence length
class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expand = nn.Linear(config["d_model"], 4 * config["d_model"])
        self.gelu = nn.GELU()
        self.reduce = nn.Linear(4 * config["d_model"], config["d_model"])

    def forward(self, x):
        x = self.expand(x)
        x = self.gelu(x)
        x = self.reduce(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config["d_model"])
        self.attn = SelfAttention(config)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.layer_norm(x))
        x = x + self.ffn(self.layer_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config["vocab_size"], config["d_model"]),
            wpe = nn.Embedding(config["seq_len"], config["d_model"]),
            h = nn.ModuleList([Block(config) for _ in range(config["num_layers"])]),
            ln_f = nn.LayerNorm(config["d_model"]),
        ))
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

def main():

    # Load hyperparameters
    with open('gpt_config.json', 'r') as f:
        hyperparameters = json.load(f)

if __name__ == "__main__":
    main() 
    
