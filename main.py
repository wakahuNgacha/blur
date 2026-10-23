from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


async def save_image(file: UploadFile, directory: Path) -> str:
    extension = IMAGE_EXTENSIONS.get(file.content_type or "")
    if extension is None:
        raise HTTPException(status_code=415, detail="Upload must be a supported image")

    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4().hex}{extension}"
    destination = directory / filename
    saved = False

    try:
        with destination.open("wb") as image:
            while chunk := await file.read(1024 * 1024):
                image.write(chunk)
        saved = True
    finally:
        await file.close()
        if not saved:
            destination.unlink(missing_ok=True)

    return filename


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/faceProfile")
async def read_face_profile(file: UploadFile = File(...)):
    filename = await save_image(file, BASE_DIR / "faceProfiles")
    return {"message": "Face profile saved", "filename": filename}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = await save_image(file, BASE_DIR / "uploads")
    return {"message": "Image uploaded", "filename": filename}


@app.get("/download")
def download_file():
    
    return {"message": "Download endpoint"}

