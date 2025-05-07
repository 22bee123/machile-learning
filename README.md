# Crop Disease Detection Web Application

A modern web application for detecting crop diseases using YOLOv8 and Flask. This project provides a user-friendly interface for real-time crop disease detection through your webcam.

## Features

- Live webcam feed with real-time disease detection
- Adjustable confidence threshold for detections
- Capture and analyze individual frames
- Modern, minimalist UI design
- Responsive layout for different device sizes

## Requirements

- Python 3.8 or higher
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application
   ```
   python app.py
   ```

2. Open your web browser and navigate to
   ```
   http://127.0.0.1:5000
   ```

3. Allow access to your webcam when prompted by the browser

4. Use the interface to:
   - View the live detection feed
   - Adjust the confidence threshold using the slider
   - Capture and analyze frames using the "Capture and Analyze" button

## How It Works

The application uses a YOLOv8 model trained on crop disease images. When a webcam frame is captured, the model processes the image and identifies any diseases present, highlighting them with bounding boxes and providing confidence scores.

## Development

This application is built with:
- Flask for the web server
- YOLOv8 for object detection
- HTML/CSS/JavaScript for the frontend
- OpenCV for image processing

## License

This project is licensed under the MIT License - see the LICENSE file for details. 