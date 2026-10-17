"""Load ADNI slices from a patient-level split manifest."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ADNISliceDataset(Dataset):
    """Return a grayscale image tensor and its binary class label."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)

        with self.manifest_path.open(newline="") as file:
            reader = csv.DictReader(file)
            required = {"path", "target", "patient_id", "scan_id"}

            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Manifest must contain these columns: {required}"
                )

            self.rows = list(reader)

        if not self.rows:
            raise ValueError("The manifest contains no images.")

        for row in self.rows:
            if row["target"] not in {"0", "1"}:
                raise ValueError(f"Invalid target: {row['target']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["path"]) as image:
            image = image.convert("L")
            image_tensor = TF.to_tensor(image)

        label = torch.tensor(int(row["target"]), dtype=torch.long)

        return image_tensor, label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check one image from an ADNI split manifest."
    )
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    dataset = ADNISliceDataset(args.manifest)
    image, label = dataset[0]

    print("Images in manifest:", len(dataset))
    print("Image shape [channels, height, width]:", tuple(image.shape))
    print("Image dtype:", image.dtype)
    print("Pixel range:", image.min().item(), image.max().item())
    print("Label:", label.item())
    print("All pixel values finite:", torch.isfinite(image).all().item())