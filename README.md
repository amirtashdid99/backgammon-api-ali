# Backgammon Detection API - Render.com

Flask API server with YOLOv8 model for backgammon piece detection.

## Quick Deploy to Render

1. Create new Web Service on [render.com](https://render.com)
2. Connect this repository
3. Set root directory to `deploy_render`
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Deploy!

## Environment Variables (Optional)

- `MODEL_PATH`: Path or URL to custom trained model (default: `yolov8n.pt`)

## API Endpoints

- `GET /`: API info
- `GET /health`: Health check
- `POST /detect`: Detect objects in image

## Local Testing

```bash
cd deploy_render
pip install -r requirements.txt
python app.py
```

Then test:
```bash
curl http://localhost:5000/health
```

## Using Custom Trained Model

1. Upload your `best.pt` to Google Drive/Dropbox
2. Get shareable link
3. Set `MODEL_PATH` environment variable in Render dashboard
4. Redeploy

## Performance

- Free tier: May have cold starts (~10-30s first request)
- Paid tier ($7/month): Faster, no cold starts
