import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        ff_dim=1024
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            embedding_dim,
            ff_dim
        )

        self.activation = nn.ReLU()

        self.linear2 = nn.Linear(
            ff_dim,
            embedding_dim
        )

    def forward(self, x):

        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        return x