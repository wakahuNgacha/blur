import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import face_recognition
import numpy as np

DB_NAME = "faces.db"
MODEL_NAME = "dlib_face_recognition_resnet"


def save_embedding(embedding, person_label):
    """Save one face embedding in the existing SQLite database."""
    vector = np.asarray(embedding, dtype=np.float64)

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute("""
            INSERT INTO faces (
                person_label,
                embedding,
                embedding_dimension,
                model_name,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            person_label,
            json.dumps(vector.tolist()),
            int(vector.size),
            MODEL_NAME,
            datetime.now(timezone.utc).isoformat()
        ))

        return cursor.lastrowid


def detect_and_save_faces(image_path, person_label=None,
                          show_result=True):
    """Detect faces, calculate embeddings, and save them."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr_image = cv2.imread(str(image_path))

    if bgr_image is None:
        raise ValueError("Could not read the image file.")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Detect faces using the HOG detector.
    locations = face_recognition.face_locations(
        rgb_image,
        number_of_times_to_upsample=1,
        model="hog"
    )

    if not locations:
        print("No faces detected.")
        return []

    embeddings = face_recognition.face_encodings(
        rgb_image,
        known_face_locations=locations,
        num_jitters=1,
        model="small"
    )

    saved_records = []

    for index, (location, embedding) in enumerate(
        zip(locations, embeddings), start=1
    ):
        label = person_label or (
            f"image_{image_path.stem}_face_{index}"
        )

        face_number = save_embedding(embedding, label)

        saved_records.append({
            "face_number": face_number,
            "person_label": label,
            "embedding_dimension": len(embedding)
        })

        top, right, bottom, left = location

        cv2.rectangle(
            bgr_image,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        cv2.putText(
            bgr_image,
            f"Face {face_number}",
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(
            f"Saved face #{face_number} "
            f"with {len(embedding)} embedding values."
        )

    print(f"Total faces saved: {len(saved_records)}")

    if show_result:
        cv2.imshow("Detected Faces", bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return saved_records


if __name__ == "__main__":
    image_file = (
        "faceProfiles/"
        "IMG_0258.jpg"
    )

    detect_and_save_faces(
        image_path=image_file,
        person_label=None,
        show_result=True
    )

