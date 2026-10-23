# Video Privacy Blur

An AI-powered computer vision system that automatically detects and blurs people's faces in videos to protect privacy.

## Features

- Upload and process videos
- Detect faces automatically
- Blur detected faces
- Track faces across video frames
- Export anonymized videos

## Tech Stack

- Python
- OpenCV
- YOLO
- FastAPI
- FFmpeg

## Workflow

```text
                    VIDEO
                      ↓
               Frame Extraction
                      ↓
               Face Detection
                      ↓
                Face Tracking
                      ↓
              Face Recognition
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
       WANTED                  UNWANTED
          ↓                       ↓
       SHOW                  APPLY BLUR
          ↓                       ↓
          └───────────┬───────────┘
                      ↓
                Video Encoding
                      ↓
                OUTPUT VIDEO
```

## Image upload API

The FastAPI app accepts a multipart form field named `file` at both upload
routes. `POST /faceProfile` saves the image in `faceProfiles/`, while
`POST /upload` saves it in `uploads/`. Each accepted upload gets a unique
filename; supported image types are JPEG, PNG, GIF, WebP, BMP, and TIFF.

Install `python-multipart` to enable multipart form uploads, then send an image
with either route:

```bash
curl -F "file=@portrait.jpg" http://localhost:8000/faceProfile
curl -F "file=@image.png" http://localhost:8000/upload