import os
from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import Config
from src.dataset import PlanetDataset, DatasetConfig
from src.dataset_splitter import stratify_shuffle_split_subsets


class PlanetDM(LightningDataModule):
    def __init__(self, config: Config):
        """
        Initialize the PlanetDM data module.

        Args:
            config (Config): Configuration object containing parameters for the data module.
        """
        super().__init__()
        self._config = config
        self._augmentation_params = config.augmentation_params
        self._images_folder = os.path.join(self._config.data_config.data_path, 'train-jpg')
        self.label_encoder = self._config.label_encoder
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        """
        Prepare data by splitting and saving datasets.
        """
        split_and_save_datasets(self._config.data_config.data_path, self._config.seed, self._config.data_config.train_size)

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage (Optional[str]): Stage of the setup ('fit', 'test', etc.).
        """
        if stage == 'fit':
            df_train = read_df(self._config.data_config.data_path, 'train')
            df_valid = read_df(self._config.data_config.data_path, 'valid')
            config = DatasetConfig(
                image_folder=self._images_folder,
                num_classes=len(self.label_encoder),
                transforms=get_transforms(
                    aug_config=self._augmentation_params,
                    width=self._config.data_config.width,
                    height=self._config.data_config.height,
                ),
                label_encoder=self.label_encoder,
            )
            self.train_dataset = PlanetDataset(df_train, config)
            self.valid_dataset = PlanetDataset(df_valid, config)

        elif stage == 'test':
            df_test = read_df(self._config.data_config.data_path, 'test')
            config = DatasetConfig(
                image_folder=self._images_folder,
                num_classes=len(self.label_encoder),
                transforms=get_transforms(
                    aug_config=self._augmentation_params,
                    width=self._config.data_config.width,
                    height=self._config.data_config.height,
                ),
                label_encoder=self.label_encoder,
            )
            self.test_dataset = PlanetDataset(df_test, config)

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def split_and_save_datasets(data_path: str, seed: int, train_fraction: float = 0.8):
    """
    Split and save the datasets into train, validation, and test sets.

    Args:
        data_path (str): Path to the data directory.
        seed (int): Seed for reproducibility.
        train_fraction (float): Fraction of data to use for training.
    """
    df = pd.read_csv(os.path.join(data_path, 'train_classes.csv'))
    df = df.drop_duplicates()

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(df, seed, train_fraction=train_fraction)

    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """
    Read a DataFrame from a CSV file.

    Args:
        data_path (str): Path to the data directory.
        mode (str): Mode of the data ('train', 'valid', 'test').

    Returns:
        pd.DataFrame: DataFrame read from the CSV file.
    """
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
