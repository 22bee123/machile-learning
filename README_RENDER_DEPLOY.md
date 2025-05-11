# Deploying Crop Disease Detection System to Render

## 1. Prerequisites
- All code, model files, and dependencies should be committed to your Git repository.
- Your YOLOv8 model file is located at:
  `crop-disease-detection-using-yolov8/Crop Disease Detection Using YOLOv8/runs/detect/train3/weights/best.pt`
- Your main Flask app is `app.py`.

## 2. Files for Render
- `requirements.txt` (includes Flask, ultralytics, opencv-python, numpy, gunicorn, Flask-Cors)
- `render.yaml` (tells Render how to build and run your app)

## 3. Steps to Deploy

### a. Push your code to GitHub (or GitLab/Bitbucket)
- Make sure all files, including the model, are pushed. For large model files (>100MB), consider using [Git LFS](https://git-lfs.github.com/) or a download script.

### b. Create a new Web Service on Render
- Go to [https://dashboard.render.com/](https://dashboard.render.com/)
- Click "New +" > "Web Service"
- Connect your GitHub repo
- Render will auto-detect `render.yaml` and set up build & start commands
- Choose Free Plan (or paid for more resources)

### c. Environment Variables
- No extra variables needed unless you want to set custom config

### d. Wait for Build & Deploy
- Render will install dependencies, then start your app using gunicorn.
- Once deployed, you’ll get a public URL (e.g., `https://your-app.onrender.com`)

### e. Test All Features
- Visit your Render URL and test:
  - Live video feed (if using webcam, note: Render cannot access server webcam, only uploaded images will work)
  - Image upload and detection
  - Confidence threshold slider
  - Any other features

## 4. Notes & Limitations
- **Webcam access:** Render servers cannot access your local webcam. Only features that use uploaded images will work for remote users.
- **Model size:** If your model is very large, Render’s free tier may have issues. Paid plans or model download scripts can help.
- **CPU only:** Free Render services do not provide GPU. Inference will be slower than on your local machine.

## 5. Troubleshooting
- Check Render logs for errors (Deploy > Logs)
- Make sure model path is correct and model file is present after deploy
- For large files, check Render’s file size and storage limits

---

For more help, see [Render Python docs](https://render.com/docs/deploy-flask) or ask for further guidance!
