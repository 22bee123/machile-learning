import os
import requests

# TODO: Replace 'YOUR_FILE_ID_HERE' with the actual file ID for best.pt
MODEL_URL = 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE'
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
