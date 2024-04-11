"""
Created By: ishwor subedi
Date: 2024-03-29
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import os
from mltu.configs import BaseModelConfigs
from src.services.test.main_recognition import ImageToWordModel

app = FastAPI()


class ImageData(BaseModel):
    image_data: str
    access_token: str


def single_image_inference(image_path: str):
    """
    This method is used to predict the text from the image path provided.
    :param image_path:  image path for prediction
    :return:  predicted text
    """
    model_information_config = "resources/models/202403281555/configs.yaml"
    configs = BaseModelConfigs.load(model_information_config)
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    image = cv2.imread(image_path)
    prediction_text = model.predict(image)
    return prediction_text


@app.post("/predict")
async def predict_captcha(data: ImageData):
    """
    This method is used to predict the text from the image data provided in the request body.
    :param data:  image data in base64 format or image path
    :return:  predicted text
    """
    access_token = data.access_token
    try:
        img_data = base64.b64decode(data.image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite('temp_img.png', img)
        captcha_text = single_image_inference('temp_img.png')
        os.remove('temp_img.png')
    except:
        if os.path.isfile(data.image_data):
            captcha_text = single_image_inference(data.image_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid image data provided")

    return {"Captcha Text": captcha_text}
