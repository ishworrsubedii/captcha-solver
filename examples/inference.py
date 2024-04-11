"""
Created By: ishwor subedi
Date: 2024-03-28
"""

import os

import cv2

from mltu.configs import BaseModelConfigs

from src.services.test.main_recognition import ImageToWordModel
from src.services.test.save_results import save_result_text


def main(draw: bool, model_information_config, output_result_file, image_test_dir):
    """
    This method is used to predict the text from the image and save the result in the output file and display the image with the predicted text
    if draw is True else only save the result in the output file and print the result in the console.

    :param draw: draw the image with the predicted text if True else not
    :param model_information_config: model information config file
    :param output_result_file: output result file
    :param image_test_dir: image test directory
    :return:
    """
    configs = BaseModelConfigs.load(model_information_config)
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    output_text_file = output_result_file
    image_dir = image_test_dir
    for i in os.listdir(image_dir):
        image_path = os.path.join(image_dir, i)

        image = cv2.imread(image_path)

        prediction_text = model.predict(image)
        if draw:
            window_title = f" Predicted: {prediction_text}"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            width, height = 800, 600
            cv2.resizeWindow(window_title, width, height)
            cv2.imshow(window_title, image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        save_result_text(image_path=image_path, prediction_text=prediction_text, output_file=output_text_file)

        print(f"Image: {image_path}, Prediction: {prediction_text}")
        print("_____________________________________________________________________\n")


if __name__ == "__main__":
    draw = True
    model_information_config = "resources/models/202403281555/configs.yaml"
    output_result_file = "output/result.txt"
    image_test_dir = "resources/dataset/dataset_20240328/test"

    main(draw=draw, model_information_config=model_information_config, output_result_file=output_result_file,
         image_test_dir=image_test_dir)
