"""
Created By: ishwor subedi
Date: 2024-03-29
"""

import cv2

from mltu.configs import BaseModelConfigs

from src.services.test.main_recognition import ImageToWordModel


def single_image_inference(model_information_config, image_path: str):
    configs = BaseModelConfigs.load(model_information_config)
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    image = cv2.imread(image_path)

    prediction_text = model.predict(image)

    print(f"Image: {image_path}, Prediction: {prediction_text}")


if __name__ == '__main__':
    model_information_config = "resources/models/202403281555/configs.yaml"

    image_path = "resources/dataset/dataset_20240328/test/100999.png"

    single_image_inference(model_information_config=model_information_config, image_path=image_path)
