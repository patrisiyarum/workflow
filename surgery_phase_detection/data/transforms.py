"""
Data Transforms
===============

Defines image preprocessing and augmentation pipelines for training and
evaluation. Uses torchvision transforms for consistency with pretrained models.
"""

from typing import List, Optional, Tuple

from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    horizontal_flip: float = 0.5,
    rotation: int = 10,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05,
) -> transforms.Compose:
    """Build the training augmentation and preprocessing pipeline.

    Args:
        image_size: Target image size (square).
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).
        horizontal_flip: Probability of random horizontal flip.
        rotation: Max rotation in degrees.
        brightness: ColorJitter brightness factor.
        contrast: ColorJitter contrast factor.
        saturation: ColorJitter saturation factor.
        hue: ColorJitter hue factor.

    Returns:
        A composed transform pipeline for training data.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomRotation(degrees=rotation),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
    )


def get_val_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Build the validation/test preprocessing pipeline (no augmentation).

    Args:
        image_size: Target image size (square).
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).

    Returns:
        A composed transform pipeline for validation/test data.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_inverse_normalize(
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Normalize:
    """Return the inverse normalization transform for visualization.

    Useful for converting a normalized tensor back to displayable values.

    Args:
        mean: Original normalization mean.
        std: Original normalization std.

    Returns:
        An inverse normalization transform.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]

    return transforms.Normalize(mean=inv_mean, std=inv_std)
