"""
Created By: ishwor subedi
Date: 2024-03-27
"""
import os.path

import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
import typing


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray) -> str:
        """
         This method is used to predict the text from the image
        :param image: image array that need to be passed for prediction
        :return: predicted text
        """
        resized_image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(resized_image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]

        return text
