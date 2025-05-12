import os
import requests

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
        print(f"Downloading model to {destination}...")
        download_file_from_google_drive(file_id, destination)
        print("Model downloaded successfully!")
    else:
        print("Model file already exists. Skipping download.")
