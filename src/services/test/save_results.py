"""
Created By: ishwor subedi
Date: 2024-03-27
"""


def save_result_text(image_path: str, prediction_text: str, output_file: str) -> None:
    """

     Save the result of the prediction in a text file
    :param image_path: image path that is being predicted
    :param label:  actual label of the image
    :param prediction_text:  predicted label of the image
    :param output_file: output file path
    :return:
    """

    with open(output_file, 'a') as f:
        f.write(f"Image: {image_path}, Prediction: {prediction_text}\n")
