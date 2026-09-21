import torch
import torch.nn as nn
import math
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, input_features):
        y_final = []
        for i in range(len(input_features)):
            temp = []
            for j in range(len(input_features)):
                temp.append(torch.dot(input_features[i],input_features[j]))
            w = torch.softmax(torch.stack(temp)/math.sqrt(self.embedding_dim), dim=0)
            y = torch.zeros_like(input_features[i])
            for j in range(len(input_features)):
                y = y + w[j]*input_features[j]
            y_final.append(y)
        return torch.stack(y_final)
