"""
Created By: ishwor subedi
Date: 2024-03-28
"""
import cv2
import pandas as pd
from mltu.configs import BaseModelConfigs
from tqdm import tqdm

from src.services.test.accuracy_calculation import MetricsCalculator
from src.services.test.main_recognition import ImageToWordModel

if __name__ == "__main__":
    configs = 'resources/models/202403281555/configs.yaml'
    image_information = 'resources/models/202403281555/train.csv'
    configs = BaseModelConfigs.load(configs)
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    df = pd.read_csv(image_information).values.tolist()
    predictions = []
    labels = []

    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))
        prediction_text = model.predict(image)
        predictions.append(prediction_text)
        labels.append(str(label))

    metrics_calculator = MetricsCalculator(predictions, labels)
    accuracy = metrics_calculator.calculate_accuracy()
    average_cer = metrics_calculator.calculate_cer()

    print(f"Accuracy: {accuracy}")
    print(f"Average CER: {average_cer}")
