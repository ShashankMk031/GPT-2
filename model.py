import torch 
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):  # Fix 1: Changed nn.model to nn.Module
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Fix 2: Changed nn.Embeding to nn.Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, embed_size))  # Fix 3: Changed nn.Paraneter to nn.Parameter
        self.transformer_blocks = nn.ModuleList([TransformerBlocks(embed_size, num_heads) for _ in range(num_layers)])  # Fix 4: Changed TransformerBlock to TransformerBlocks
        self.ln_f = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape 
        x = self.embed(x) + self.pos_embed[:, :seq_len, :]
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.fc_out(x)  # Fix 5: Changed logists to logits
        return logits
    
class TransformerBlocks(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.ln2 = nn.LayerNorm(embed_size)
        
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x