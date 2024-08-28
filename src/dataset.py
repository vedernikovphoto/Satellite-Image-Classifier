import os
from typing import Optional
import pandas as pd
import numpy as np
import cv2
from dataclasses import dataclass
from torch.utils.data import Dataset
import albumentations as albu


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset parameters.

    Attributes:
        image_folder (str): Path to the folder containing images.
        num_classes (int): Number of classes in the dataset.
        transforms (Optional[albu.BaseCompose]): Optional albumentations transforms to be applied.
        label_encoder (Optional[dict]): Optional label encoder for the dataset labels.
    """
    image_folder: str
    num_classes: int
    transforms: Optional[albu.BaseCompose] = None
    label_encoder: Optional[dict] = None


class PlanetDataset(Dataset):
    """
    Custom Dataset class for loading and transforming images and labels.

    Args:
        df (pd.DataFrame): DataFrame containing image names and labels.
        config (DatasetConfig): Configuration object containing dataset parameters.
    """
    def __init__(self, df: pd.DataFrame, config: DatasetConfig) -> None:
        self.image_folder = config.image_folder
        self.transforms = config.transforms
        self.label_encoder = config.label_encoder
        self.num_classes = config.num_classes

        # Convert DataFrame columns to NumPy arrays
        image_names = df['image_name'].to_numpy()
        image_paths = [os.path.join(self.image_folder, f'{name}.jpg') for name in image_names]  # noqa: WPS221
        self.image_paths = np.array(image_paths)
        self.labels = df.drop('image_name', axis=1).to_numpy().astype('float32')

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get an item by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, labels) where image is the transformed image and labels are the corresponding labels.
        """
        image_path = self.image_paths[idx]
        labels = self.labels[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        return image, labels

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.image_paths)
