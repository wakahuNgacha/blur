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