from typing import List, Tuple
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
from skmultilearn.model_selection.iterative_stratification import IterativeStratification


def _prepare_data(annotation: pd.DataFrame) -> Tuple[np.array, np.array, MultiLabelBinarizer]:
    """
    Prepare data by extracting features and labels and binarizing the labels.

    Args:
        annotation (pd.DataFrame): DataFrame containing image annotations with image names and tags.

    Returns:
        Tuple[np.ndarray, np.ndarray, MultiLabelBinarizer]: Features, binarized labels, and the binarizer.
    """
    x_columns = ['image_name']
    y_columns = ['tags']
    all_x = annotation[x_columns].to_numpy()
    all_y = _split_tags(annotation[y_columns])

    mlb = MultiLabelBinarizer()
    all_y = mlb.fit_transform(all_y)

    return all_x, all_y, mlb


def _split_tags(y_data: pd.DataFrame) -> List[List[str]]:
    """
    Split tags string into a list of tags for each image.

    Args:
        y_data (pd.DataFrame): DataFrame containing the tags.

    Returns:
        List[List[str]]: List of lists of tags.
    """
    return y_data['tags'].apply(lambda tags: tags.split()).tolist()


def _get_split_data(all_x, all_y, first_indexes, second_indexes) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:    # noqa: WPS221, E501
    """
    Split data into two sets based on provided indexes.

    Args:
        all_x (np.ndarray): Array of features.
        all_y (np.ndarray): Array of labels.
        first_indexes (np.ndarray): Indexes for the first split.
        second_indexes (np.ndarray): Indexes for the second split.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split features and labels.
    """
    x_first, x_second = all_x[first_indexes], all_x[second_indexes]
    y_first, y_second = all_y[first_indexes], all_y[second_indexes]

    return x_first, y_first, x_second, y_second


def _create_subset(x_data: np.array, y_data: np.array, mlb: MultiLabelBinarizer) -> pd.DataFrame:
    """
    Create a DataFrame subset from features and labels.

    Args:
        x_data (np.ndarray): Array of features.
        y_data (np.ndarray): Array of labels.
        mlb (MultiLabelBinarizer): Fitted binarizer for label encoding.

    Returns:
        pd.DataFrame: DataFrame containing the subset.
    """
    x_columns = ['image_name']
    concatenated_data = np.concatenate([x_data, y_data], axis=1)
    column_names = x_columns + mlb.classes_.tolist()

    return pd.DataFrame(data=concatenated_data, columns=column_names)


def _split(
    xs: np.array,
    ys: np.array,
    seed: int,
    distribution: List[float],
) -> Tuple[np.array, np.array]:
    """
    Perform iterative stratification to split the data.

    Args:
        xs (np.ndarray): Array of features.
        ys (np.ndarray): Array of labels.
        seed (int): Seed for reproducibility.
        distribution (List[float]): Distribution for the split.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indexes for the two splits.
    """
    np.random.seed(seed)
    stratifier = IterativeStratification(n_splits=2, sample_distribution_per_fold=distribution)

    return next(stratifier.split(X=xs, y=ys))


def stratify_shuffle_split_subsets(
    annotation: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratify, shuffle, and split the dataset into train, validation, and test subsets.

    Args:
        annotation (pd.DataFrame): DataFrame containing image annotations with image names and tags.
        seed (int): Seed for reproducibility.
        train_fraction (float): Portion of data for training; the rest is equally split into validation and test sets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test subsets as DataFrames.
    """
    all_x, all_y, mlb = _prepare_data(annotation)

    train_indexes, else_indexes = _split(all_x, all_y, seed, distribution=[1 - train_fraction, train_fraction])  # noqa: WPS221, E501
    x_train, y_train, x_else, y_else = _get_split_data(all_x, all_y, train_indexes, else_indexes)

    test_indexes, valid_indexes = _split(x_else, y_else, seed, distribution=[0.5, 0.5])
    x_test, y_test, x_valid, y_valid = _get_split_data(x_else, y_else, test_indexes, valid_indexes)

    train_subset = _create_subset(x_train, y_train, mlb)
    valid_subset = _create_subset(x_valid, y_valid, mlb)
    test_subset = _create_subset(x_test, y_test, mlb)

    return train_subset, valid_subset, test_subset
