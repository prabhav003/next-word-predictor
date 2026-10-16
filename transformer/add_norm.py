import torch.nn as nn


class AddAndNormalize(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.layer_norm = nn.LayerNorm(
            embedding_dim
        )

    def forward(
        self,
        x,
        sublayer_output
    ):

        # Residual connection.
        x = x + sublayer_output

        # Normalize the final dimension.
        return self.layer_norm(x)