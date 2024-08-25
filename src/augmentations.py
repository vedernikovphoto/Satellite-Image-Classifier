import albumentations as albu
from albumentations.pytorch import ToTensorV2
from src.config import AugmentationConfig
from dataclasses import dataclass


@dataclass
class TransformFlags:
    """
    Flags to enable or disable different types of transformations.

    Attributes:
        preprocessing (bool): If True, apply preprocessing transformations.
        augmentations (bool): If True, apply augmentation transformations.
        postprocessing (bool): If True, apply postprocessing transformations.
    """
    preprocessing: bool = True
    augmentations: bool = True
    postprocessing: bool = True


def get_transforms(
    aug_config: AugmentationConfig,
    width: int,
    height: int,
    flags: TransformFlags = None,
) -> albu.Compose:
    """
    Get the data augmentation and preprocessing transformations.

    Args:
        aug_config (AugmentationConfig): Augmentation configuration object.
        width (int): Width to resize the image to.
        height (int): Height to resize the image to.
        flags (TransformFlags): Transform flags for preprocessing, augmentations, and postprocessing.

    Returns:
        albu.BaseCompose: Composed transformations.
    """
    if flags is None:
        flags = TransformFlags()
    transforms = []

    if flags.preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if flags.augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.HueSaturationValue(
                    hue_shift_limit=aug_config.hue_shift_limit,
                    sat_shift_limit=aug_config.sat_shift_limit,
                    val_shift_limit=aug_config.val_shift_limit,
                    p=0.5,
                ),
                albu.RandomBrightnessContrast(
                    brightness_limit=aug_config.brightness_limit,
                    contrast_limit=aug_config.contrast_limit,
                    p=0.5,
                ),
                albu.ShiftScaleRotate(),
                albu.GaussianBlur(),
            ],
        )

    if flags.postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
