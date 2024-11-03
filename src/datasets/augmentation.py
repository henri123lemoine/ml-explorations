import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms


def get_advanced_transforms(train: bool = True):
    """Enhanced augmentation pipeline using albumentations"""
    if train:
        return A.Compose(
            [
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.OneOf(
                    [
                        A.GaussNoise(p=1),
                        A.GaussianBlur(p=1),
                        A.MotionBlur(p=1),
                    ],
                    p=0.2,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


class TrainTransform:
    """Picklable transform class for training data augmentation"""

    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Convert numpy array to PIL Image, apply transforms, convert back
        pil_image = Image.fromarray(image)
        transformed = self.transform(pil_image)
        return np.array(transformed)
