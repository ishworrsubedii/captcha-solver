"""
Created By: ishwor subedi
Date: 2024-03-27
"""
import stow
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    """
    This class is used to define the models configuration
    """

    def __init__(self):
        super().__init__()
        self.model_path = stow.join('resources/models/',
                                    datetime.strftime(datetime.now(), "%Y%m%d%H%M"))  # Path to save the models
        self.dataset_path = 'train_images/dataset/images'  # Path to the dataset
        self.splitted_dataset_path = 'resources/dataset'  # Path to the splitted dataset
        self.vocab = ''  # Vocabulary for the models
        self.height = 50  # Height of the image
        self.width = 200  # Width of the image
        self.max_text_length = 0  # Maximum length of the text in the dataset
        self.batch_size = 200  # Batch size for training
        self.learning_rate = 1e-2  # Learning rate for the models
        self.train_epochs = 1# Number of epochs to train the models
        self.train_workers = 20  # Number of workers to use for training
