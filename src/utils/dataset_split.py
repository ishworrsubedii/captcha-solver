"""
Created By: ishwor subedi
Date: 2024-03-28
"""
from sklearn.model_selection import train_test_split
import os
from src.entity.configuration import ModelConfigs

from sklearn.model_selection import train_test_split
import os
import shutil
from datetime import datetime


def split_dataset_into_train_and_test(dataset_path, destination_path, test_size=0.2):
    """
    Splits the dataset into training and testing sets and saves them in the destination path.

    Parameters:
    - dataset_path: The path to the dataset.
    - destination_path: The path where the split datasets will be saved.
    - test_size: The proportion of the dataset to include in the test split (default is 0.2).

    Returns:
    - train_dir: The directory of the training set.
    - test_dir: The directory of the testing set.
    """
    # Get all files in the dataset
    all_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]

    # Split the files into training and testing sets
    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)

    # Create directories for the training and testing sets
    today_date = datetime.today().strftime('%Y%m%d')
    train_dir = os.path.join(destination_path, f"dataset_{today_date}", "train")
    test_dir = os.path.join(destination_path, f"dataset_{today_date}", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(file, train_dir)

    for file in test_files:
        shutil.copy(file, test_dir)

    print(
        f"{len(train_files)} training images and {len(test_files)} testing images.")
