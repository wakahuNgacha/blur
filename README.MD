# Face Blur

Face Blur is a local Python application for processing a group photo against a
single reference face. It detects faces in both images, compares their face
embeddings, leaves faces matching the reference visible, and blurs the other
detected faces. A FastAPI endpoint accepts the images and provides a download
for the processed photo.

This repository processes **still images**, not videos. It does not use YOLO,
face tracking, or FFmpeg.

## What it contains

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI application: image uploads, processing endpoint, and download endpoint. |
| `faceBlur.py` | Main image pipeline: face detection, matching, blurring, and output writing. |
| `dbcreator.py` | Creates the SQLite `faces` table used by the separate database utilities. |
| `faceDetector.py` | Detects faces in an image, stores their embeddings in SQLite, and can display an annotated image. |
| `faceRecognizer.py` | Compares image faces with stored database embeddings and writes an annotated result image. |

The API uses `faceBlur.py`; it does not read or write the SQLite database.
`faceDetector.py` and `faceRecognizer.py` are separate command-line utilities.

## Requirements and setup

Use Python with working installations of OpenCV, `face_recognition` (and its
`dlib` dependency), NumPy, FastAPI, Uvicorn, and `python-multipart`. The project
does not currently include a dependency manifest or pin dependency versions.
Installing `dlib` can require native build tools, depending on your operating
system and Python version.

From the project root, create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn python-multipart opencv-python face-recognition numpy
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run the API

Start the development server from the project root:

```bash
python -m uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API
documentation is available at `http://127.0.0.1:8000/docs`.

### Endpoints

#### `GET /`

Health check. Returns:

```json
{"message": "Blur API is running"}
```

#### `POST /process-images`

Submit a `multipart/form-data` request with both required fields:

| Field | Contents |
| --- | --- |
| `face_profile` | Reference image containing exactly one detectable face. |
| `group_image` | Group image containing at least one detectable face. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/process-images \
  -F "face_profile=@portrait.jpg" \
  -F "group_image=@group.jpg"
```

The upload MIME types accepted by the API are JPEG, PNG, GIF, WebP, BMP, and
TIFF. Files are decoded by OpenCV after upload; a supported MIME type does not
guarantee that a particular file can be decoded.

On success, the JSON response includes saved file names and paths, a summary,
and one result per detected group face. For example:

```json
{
  "message": "Images processed successfully",
  "face_profile": {
    "filename": "generated-name.jpg",
    "path": "faceProfiles/generated-name.jpg"
  },
  "group_image": {
    "filename": "generated-name.jpg",
    "path": "uploads/generated-name.jpg"
  },
  "processed_image": {
    "filename": "processed_image.jpg",
    "path": "processed/processed_image.jpg",
    "download_url": "/download"
  },
  "summary": {
    "faces_detected": 2,
    "faces_matched": 1,
    "faces_blurred": 1
  },
  "results": [
    {
      "face": 1,
      "match": true,
      "distance": 0.31,
      "blurred": false
    },
    {
      "face": 2,
      "match": false,
      "distance": 0.68,
      "blurred": true
    }
  ]
}
```

The distances above are illustrative. If an embedding cannot be calculated
for a detected face, that face is blurred and its result has `"distance": null`.
Processing errors return HTTP 400 for invalid or unreadable image content, 415
for an unsupported upload MIME type, and 500 for other processing failures.

#### `GET /download`

Downloads the most recently written output as `processed_image.jpg`. It returns
HTTP 404 until an image has been processed.

### Browser access

CORS is configured for `http://localhost:5173` and
`http://127.0.0.1:5173`. Other browser origins are not allowed by the current
configuration.

## How image processing works

1. Both images are loaded and converted from OpenCV's BGR format to RGB for
   `face_recognition`.
2. The reference image must contain exactly one face detected with the HOG
   detector.
3. The application calculates the reference face embedding, then detects and
   encodes each face in the group image.
4. It computes the Euclidean distance between each group-face embedding and the
   reference embedding. A distance at or below `MATCH_TOLERANCE` (`0.50`) is a
   match. The matching face is left clear; non-matching faces are blurred. A
   face whose embedding fails is blurred as a precaution.
5. The processed image is written as `processed/processed_image.jpg`, and the
   per-face results are returned by the API.

The detector uses the `face_recognition` HOG model and one upsample. Embeddings
use the library's `"small"` encoding model and one jitter. Blur regions are
expanded by 20% in each direction and processed with a Gaussian kernel of
`99 × 99`.

`MATCH_TOLERANCE` and `BLUR_KERNEL` are constants in `faceBlur.py`. A lower
matching tolerance is stricter. Recognition thresholds are not universal:
evaluate changes against representative images before relying on them.

## Optional SQLite utilities

The database tools are independent of the upload API:

1. From the project root, run `python dbcreator.py` to create `faces.db` and its
   `faces` table if the table does not already exist.
2. In `faceDetector.py`, change the example image path at the bottom of the
   file to an existing image, then run `python faceDetector.py`. Detected face
   embeddings are stored with a label and model name. The display window can
   be disabled by passing `show_result=False` when calling
   `detect_and_save_faces`.
3. In `faceRecognizer.py`, change its example image path and run
   `python faceRecognizer.py` to compare that image against saved embeddings.
   The annotated image is written next to the input image with
   `_match_results.jpg` appended to its base name.

The table stores each embedding as JSON text along with its dimension, model
name, label, and creation timestamp. The recognizer considers compatible
embeddings from `dlib_face_recognition_resnet`, selects the closest candidate,
and applies its own `MATCH_TOLERANCE` of `0.50`. The database utilities resolve
`faces.db` relative to the process's current working directory, so run them
from the project root to use the expected location.

## Files created at runtime

Paths are relative to the project root:

| Path | Contents |
| --- | --- |
| `faceProfiles/` | Uploaded reference images. |
| `uploads/` | Uploaded group images. |
| `processed/processed_image.jpg` | Processed output image; overwritten by the next successful API request. |
| `faces.db` | Optional face-embedding database used by the database utilities. |

The upload directories are created when needed. These files are local to the
server and are not automatically removed.

## Privacy and operational limitations

- Face embeddings are biometric data. Handle source images, output images, and
  database records accordingly; obtain appropriate consent and follow
  applicable privacy requirements.
- The API has no authentication, authorization, upload-size limit, or
  per-user output isolation. Do not expose it publicly as-is.
- All API requests write to the same output path. Concurrent requests can
  overwrite one another's output, and a later request replaces the image
  available from `/download`.
- Uploads and database records persist on disk until removed manually.
- Face detection and embedding comparison can miss faces or produce incorrect
  matches. A blurred result is not a guarantee of anonymization, especially
  when the face is partly visible or the image contains other identifying
  information.

## Development notes

There are no automated tests or dependency lock/manifest files in the
repository at present. The checked-in scripts and local image/database files
are not a substitute for testing the API with your own representative images.
