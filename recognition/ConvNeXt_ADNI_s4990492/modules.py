"""Neural network models for ADNI slice classification."""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A compact CNN baseline for single-channel MRI slices."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # [B, 1, 240, 256] -> [B, 16, 120, 128]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # -> [B, 32, 60, 64]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # -> [B, 64, 30, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 2)

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.classifier(flattened)

class ConvNeXtBlock(nn.Module):
    """A ConvNeXt block with layer scaling and no stochastic depth."""

    def __init__(self, channels, layer_scale_init=1e-6):
        super().__init__()

        # Apply a separate spatial filter to each channel.
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )

        # LayerNorm will operate on the last dimension.
        self.norm = nn.LayerNorm(channels, eps=1e-6)

        # Mix information across channels.
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)

        # Start the residual adjustment with a small scale.
        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(channels)
        )

    def forward(self, x):
        residual = x

        x = self.depthwise(x)

        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        x = self.layer_scale * x

        # Return to [B, C, H, W].
        x = x.permute(0, 3, 1, 2)

        return residual + x

if __name__ == "__main__":
    torch.manual_seed(42)

    model = SimpleCNN()
    model.eval()

    # Synthetic inputs only: this checks the model's output shape.
    example_images = torch.randn(2, 1, 240, 256)

    with torch.no_grad():
        logits = model(example_images)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all().item()

    print("Input shape:", tuple(example_images.shape))
    print("Output shape:", tuple(logits.shape))
    print("Parameter count:", parameter_count)
    print("Forward check passed")