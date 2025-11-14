#  Backgammon Detection API

Flask API with YOLOv8 - Optimized for Render.com & Railway.app

##  Features
-  Custom YOLOv8 model (93.5% mAP)
-  Auto-download from Google Drive
-  Fast inference (~100ms)
-  CORS enabled

##  Deploy
**Railway.app**: New Project  GitHub  Deploy
**Render.com**: New Service  GitHub  Deploy

##  Endpoints
- `GET /health` - Status check
- `POST /detect` - Image detection

##  Local Test
```bash
pip install -r requirements.txt
python app.py
```
