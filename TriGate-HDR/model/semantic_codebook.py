import torch
import torch.nn as nn

class SemanticCodebook(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, semseg):
        embedded = self.embedding(semseg)
        embedded = embedded.permute(0, 3, 1, 2).contiguous()
        return embedded
