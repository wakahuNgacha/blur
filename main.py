from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

# Import the existing processing function.
from faceBlur import process_images


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

PROFILE_DIR = BASE_DIR / "faceProfiles"
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

# One output image for this simple, single-user workflow.
OUTPUT_IMAGE = PROCESSED_DIR / "processed_image.jpg"

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
        raise HTTPException(
            status_code=415,
            detail="Upload must be a supported image."
        )

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
    return {"message": "Blur API is running"}


@app.post("/process-images")
async def upload_images(
    face_profile: UploadFile = File(...),
    group_image: UploadFile = File(...)
):
    profile_filename = None
    group_filename = None

    try:
        # 1. Save the reference image.
        profile_filename = await save_image(
            face_profile,
            PROFILE_DIR
        )

        # 2. Save the group image.
        group_filename = await save_image(
            group_image,
            UPLOAD_DIR
        )

        reference_path = PROFILE_DIR / profile_filename
        group_path = UPLOAD_DIR / group_filename

        # 3. Run face detection, recognition and blurring.
        # Run the CPU-heavy function in a worker thread.
        results = await run_in_threadpool(
            process_images,
            reference_path,
            group_path,
            OUTPUT_IMAGE
        )

        # 4. Return processing results to the frontend.
        return {
            "message": "Images processed successfully",
            "face_profile": {
                "filename": profile_filename,
                "path": f"faceProfiles/{profile_filename}"
            },
            "group_image": {
                "filename": group_filename,
                "path": f"uploads/{group_filename}"
            },
            "processed_image": {
                "filename": OUTPUT_IMAGE.name,
                "path": "processed/processed_image.jpg",
                "download_url": "/download"
            },
            "summary": {
                "faces_detected": len(results),
                "faces_matched": sum(
                    1 for item in results if item["match"]
                ),
                "faces_blurred": sum(
                    1 for item in results if item["blurred"]
                )
            },
            "results": results
        }

    except HTTPException:
        raise

    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        print(f"Image processing error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the images."
        ) from exc


@app.get("/download")
def download_file():
    # 5. Send the processed image to the browser.
    if not OUTPUT_IMAGE.is_file():
        raise HTTPException(
            status_code=404,
            detail="No processed image available. Upload and process images first."
        )

    return FileResponse(
        path=OUTPUT_IMAGE,
        media_type="image/jpeg",
        filename="processed_image.jpg"
    )


