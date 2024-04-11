"""
Created By: ishwor subedi
Date: 2024-03-29
"""
import requests
import base64


def send_request(image_data):
    try:
        with open(image_data, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('ascii')
    except:
        encoded_string = image_data

    url = "http://185.52.1.199:8000/predict"
    headers = {"Content-Type": "application/json"}
    data = {"image_data": encoded_string}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Request failed with status code {response.status_code}")


image_path = "D:/xampp/htdocs/testing/preyash/captcha/104152.png"
send_request(image_path)
if __name__ == '__main__':
    # Test with image path
    image_path = "/home/ishwor/Desktop/captcha/resources/dataset/dataset_20240328/test/102692.png"
    send_request(image_path)

    # Test with base64 encoded image
    # with open(image_path, "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    # send_request(encoded_string)
