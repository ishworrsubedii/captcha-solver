"""
Created By: ishwor subedi
Date: 2024-03-28
"""
from src.entity.configuration import ModelConfigs
from src.utils.dataset_split import split_dataset_into_train_and_test

if __name__ == '__main__':
    configs = ModelConfigs()
    split_dataset_into_train_and_test(dataset_path=configs.dataset_path, test_size=0.2,
                                      destination_path=configs.splitted_dataset_path)
