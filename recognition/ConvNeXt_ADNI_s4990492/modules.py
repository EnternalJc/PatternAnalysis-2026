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