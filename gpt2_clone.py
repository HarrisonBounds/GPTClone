import json
import torch.nn as nn


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