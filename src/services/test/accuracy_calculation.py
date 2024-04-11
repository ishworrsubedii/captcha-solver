"""
Created By: ishwor subedi
Date: 2024-03-27
"""
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.services.test.main_recognition import ImageToWordModel
from mltu.utils.text_utils import get_cer
from mltu.configs import BaseModelConfigs


class MetricsCalculator:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels

    def calculate_accuracy(self):
        correct_predictions = sum(p == l for p, l in zip(self.predictions, self.labels))
        return correct_predictions / len(self.labels)

    def calculate_cer(self):
        cer = [get_cer(p, l) for p, l in zip(self.predictions, self.labels)]
        return np.average(cer)
