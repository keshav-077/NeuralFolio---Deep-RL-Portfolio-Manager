# custom_policy.py 

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that uses a Transformer Encoder.

    It takes a flattened observation (window_size * n_features_per_step) and processes
    it as a sequence.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,  # The final output dimension
        n_features_per_step: int = 8,  # <--- CRITICAL CHANGE: Matches 5 assets + 3 macro
        window_size: int = 30,
        d_model: int = 64,       # Transformer's internal embedding dimension
        n_head: int = 4,         # Number of attention heads
        n_layers: int = 2,       # Number of transformer encoder layers
        dropout: float = 0.1
    ):

        super().__init__(observation_space, features_dim)

        self.window_size = window_size
        self.n_features_per_step = n_features_per_step

        # Input shape check
        expected_flat_dim = window_size * n_features_per_step
        if observation_space.shape[0] != expected_flat_dim:
            raise ValueError(
                f"Observation space flat dimension {observation_space.shape[0]} "
                f"does not match expected {expected_flat_dim} "
                f"(window_size={window_size}, n_features_per_step={n_features_per_step})."
            )

        # 1. Input Projection:
        self.input_projection = nn.Linear(n_features_per_step, d_model)

        # 2. Positional Encoding:
        self.positional_encoding = nn.Parameter(torch.randn(1, window_size, d_model))

        # 3. Transformer Encoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Output Layers:
        self.flatten = nn.Flatten()
        self.linear_out = nn.Linear(window_size * d_model, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, window_size * n_features_per_step)

        # 1. Reshape to (batch_size, window_size, n_features_per_step)
        x = observations.reshape(-1, self.window_size, self.n_features_per_step)

        # 2. Project input features to d_model
        x = self.input_projection(x)

        # 3. Add positional encoding
        x = x + self.positional_encoding

        # 4. Pass through Transformer
        x = self.transformer_encoder(x)

        # 5. Flatten and project to final output
        x = self.flatten(x)
        x = self.relu(self.linear_out(x))

        return x