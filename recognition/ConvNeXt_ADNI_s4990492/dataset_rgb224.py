"""Identical input processing for the two initialization experiments."""
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from dataset import ADNISliceDataset


class RGB224Dataset(ADNISliceDataset):
    def __init__(self, manifest_path, training=False):
        super().__init__(manifest_path)
        self.training = training
        self.affine = transforms.RandomAffine(5, translate=(0.03, 0.03),
                                             interpolation=InterpolationMode.BILINEAR, fill=0)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as source:
            image = ImageOps.pad(source.convert('L'), (224, 224),
                                 method=Image.Resampling.BILINEAR, color=0)
            if self.training:
                image = self.affine(image)
            image = TF.to_tensor(image.convert('RGB'))
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor(int(row['target']), dtype=torch.long)
