import os
import requests

import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    file_id = "11VI1oY4g-EF0QFZWnQ5XhNFIu470_N5X"  # best.pt file ID from Google Drive
    destination = os.path.join("crop-disease-detection-using-yolov8", "Crop Disease Detection Using YOLOv8", "runs", "detect", "train3", "weights", "best.pt")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        print("Downloading model...")
        download_file_from_google_drive(file_id, destination)
        print("Model downloaded!")
    else:
        print("Model already exists.")

MODEL_PATH = os.path.join('crop-disease-detection-using-yolov8', 'Crop Disease Detection Using YOLOv8', 'runs', 'detect', 'train3', 'weights', 'best.pt')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def download_file_from_google_drive(url, destination):
    print(f"Downloading model from {url} ...")
    session = requests.Session()
    response = session.get(url, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print(f"Model downloaded to {destination}")

if not os.path.exists(MODEL_PATH):
    download_file_from_google_drive(MODEL_URL, MODEL_PATH)
else:
    print('Model already exists.')
