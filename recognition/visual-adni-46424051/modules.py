##################################   modules.py   ##################################

import torch
from torch.nn import Module, Linear, Parameter, ModuleList, Softmax, LayerNorm, Sequential, GELU
from utils import patch, position

class Model(Module):
    def __init__(self, shape=(1, 105, 105), patches=7, hidden_dim=8, blocks=2, heads=2, out_dim=10):
        super(Model, self).__init__()
        self.shape = shape
        self.patches = patches
        self.hidden_dim = hidden_dim
        self.blocks = blocks
        self.out_dim = out_dim

        self.input_dim = int(shape[0] * self.patches ** 2)
        self.linear = Linear(self.input_dim, self.hidden_dim)

        self.token = Parameter(torch.rand(1, self.hidden_dim))

        self.pos = Parameter(torch.tensor(position(self.patches ** 2 + 1, self.hidden_dim)))
        self.pos.requires_grad = False

        self.block = ModuleList([Block(self.hidden_dim, self.heads) for _ in range(self.blocks)])

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.out_dim),
            Softmax(dim=-1)
        )
    
    def forward(self, img):
        patches = patch(img, self.patches)
        
        tokens = self.linear(patches)
        tokens = torch.stack([torch.vstack((self.token, tokens[i])) for i in range(len(tokens))])
        
        out = tokens + self.pos

        for block in self.block:
            out = block(out)

        return self.mlp(out[:,0])
    
class MSA(Module):
    def __init__(self, dim, heads=2):
        super(MSA, self).__init__()
        self.dim = dim
        self.heads = heads

        self.head_dim = int(self.dim / self.heads)

        self.qmap = ModuleList([Linear(self.head_dim, self.head_dim) for _ in range(self.heads)])
        self.kmap = ModuleList([Linear(self.head_dim, self.head_dim) for _ in range(self.heads)])
        self.vmap = ModuleList([Linear(self.head_dim, self.head_dim) for _ in range(self.heads)])

        self.softmax = Softmax(dim=-1)

    def forward(self, seq):
        result = []
        for head in range(self.heads):
            qmap = self.qmap[head]
            kmap = self.kmap[head]
            vmap = self.vmap[head]

            s = seq[:, head * self.head_dim: (head + 1) * self.head_dim]
            q, k, v = qmap(s), kmap(s), vmap(s)

            result.append(self.softmax(q @ k.T / (self.head_dim ** 0.5)) @ v)
        return torch.hstack(result)

class Block(Module):
    def __init__(self, hidden_dim, heads, ratio=4):
        super(Block, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.norm = LayerNorm(hidden_dim)
        self.msa = MSA(hidden_dim, heads)
        self.norm1 = LayerNorm(self.hidden_dim)
        self.mlp = Sequential(
            Linear(self.hidden_dim, ratio * self.hidden_dim),
            GELU(),
            Linear(ratio * self.hidden_dim, self.hidden_dim)
        )

    def forward(self, img):
        out = img + self.msa(self.norm(img))
        return out + self.mlp(self.norm1(out))