import json
import torch.nn as nn
import torch
import math
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken

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
        self.c_attn = nn.Linear(config["d_model"], 3 * config["d_model"]) 

        #linear layer for output projection
        self.c_proj = nn.Linear(config["d_model"], config["d_model"])
        self.c_proj.SCALE_INIT = 1

        #mask to prevent seeing future token (lower triangle)
        self.register_buffer("bias", torch.tril(torch.ones(config["seq_len"],  config["seq_len"])).view(1, 1, config["seq_len"], config["seq_len"]))

        self.num_heads = config["num_heads"]
        self.d_model = config["d_model"]
        self.seq_len = config["seq_len"] # Store sequence length

    def forward(self, x):
        #Batch size, Sequence length, and Embedding dimension
        B, T, C = x.size()

        # Apply the combined linear layer to get query, key, and value projections
        # Shape: (B, T, 3 * d_model)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        #Reshape tensors to prepare for multi head attention
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        #Calculate attention score
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

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
        
class MLP(nn.Module):
    """
    The Feed-Forward Network (FFN) within each Transformer block.

    It processes each token's representation independently, first expanding
    the dimensionality to capture more complex features, applying a non-linear
    activation (GELU), and then reducing the dimensionality back to the
    original size.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["d_model"], 4 * config["d_model"])
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config["d_model"], config["d_model"])
        self.c_proj.SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
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
        self.ln_1 = nn.LayerNorm(config["d_model"])
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["d_model"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config["vocab_size"], config["d_model"]),
            wpe = nn.Embedding(config["seq_len"], config["d_model"]),
            h = nn.ModuleList([Block(config) for _ in range(config["num_layers"])]),
            ln_f = nn.LayerNorm(config["d_model"]),
        ))
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.seq_len = config["seq_len"]

        #weight sharing - done in GPT2 - more efficient in training process - saves parameters
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        #Scale down standard deviation - 2 * layers comes from attention and MLP
        if hasattr(module, 'SCALE_INIT'):
            std *= (2 * self.config["num_layers"]) ** 0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.seq_len, f"Cannot forward sequence of length {T}, block size is only {self.seq_len}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, gpt_config):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
      
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['seq_len'] = 1024 # always 1024 for GPT model checkpoints
    
        config = gpt_config
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device: ", device)
    
    # Load hyperparameters
    with open('gpt_config.json', 'r') as f:
        hyperparameters = json.load(f)

    #model = GPT.from_pretrained('gpt2', hyperparameters)
    model = GPT(hyperparameters)
    model.eval()
    #model.to('cuda')

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I am a language model,")

    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(hyperparameters["num_return_sequences"], 1)
    x = tokens

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    max_length = hyperparameters.get("max_length", 50) # Use get() with a default value
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)

            xcol = torch.gather(topk_indices, -1, ix)

            x = torch.cat((x, xcol), dim=1)

    for i in range(hyperparameters["num_return_sequences"]):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

if __name__ == "__main__":
    main()
