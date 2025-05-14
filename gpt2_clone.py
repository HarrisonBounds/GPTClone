import json
import torch.nn as nn
import torch
import math
from torch.nn import functional as F

class SelfAttention(nn.Module):
    """
    The Self-Attention mechanism, allowing the model to attend to different
    parts of the input sequence when processing each token.

    It computes attention scores based on the relationships between tokens
    and uses these scores to weigh the importance of other tokens when
    representing the current token. This implementation uses multi-head
    attention for richer representations and causal masking for autoregressive
    generation.
    """
    def __init__(self, config):
        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0

        #q, k, v projections

        #linear layer to combine attention heads
        self.c_attn = self.Linear(config["d_model"], 3 * config["d_model"]) 

        #linear layer for output projection
        self.c_proj = nn.Linear(config["d_model"], config["d_model"])

        #mask to prevent seeing future token (lower triangle)
        self.register_buffer("bias", torch.tril(torch.ones(config["seq_len"],  config["seq_len"])).view(1, 1, config["seq_len"], config["seq_len"]))

        self.num_heads = config["num_heads"]
        self.d_model = config["d_model"]

    def forward(self, x):

        #Batch size, Sequence length, and Embedding dimension
        B, T, C = x.size()

        # Apply the combined linear layer to get query, key, and value projections
        # Shape: (B, T, 3 * d_model)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        #Reshape tensors to prepare for multi head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        #Calculate attention score
        att = (q @ v.transpose(-2, 1)) * (1.0 / math.sqrt(k.size(-1)))

        #Mask attention scores
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        #Get attention weights
        att = F.softmax(att, dim=-1)

        #Weighted values
        y = att @ v

        #Reshape and concatenate to prepare for output
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        #Apply output projection
        y = self.c_proj(y)

        return y
        

    


class FFN(nn.Module):
    """
    The Feed-Forward Network (FFN) within each Transformer block.

    It processes each token's representation independently, first expanding
    the dimensionality to capture more complex features, applying a non-linear
    activation (GELU), and then reducing the dimensionality back to the
    original size.
    """
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
    """
    Represents a single Transformer block.

    Each block consists of two main components:
    1. Self-Attention mechanism (`SelfAttention`).
    2. Feed-Forward Network (`FFN`).

    Residual connections are applied around both the attention and the FFN,
    and layer normalization is applied before each.
    """
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
    """
    The main GPT model architecture.

    It consists of:
    - Token Embeddings (`wte`): Converts input token IDs to dense vectors.
    - Positional Embeddings (`wpe`): Encodes the position of each token in the sequence.
    - Transformer Blocks (`h`): A stack of `Block` layers.
    - Final Layer Normalization (`ln_f`): Applies layer normalization to the output of the last block.
    - Language Model Head (`lm_head`): A linear layer that projects the final representations to the vocabulary size for language modeling.
    """
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
    """Loads hyperparameters and initializes the GPT model."""
    # Load hyperparameters
    with open('gpt_config.json', 'r') as f:
        hyperparameters = json.load(f)

    GPT(hyperparameters)

if __name__ == "__main__":
    main()