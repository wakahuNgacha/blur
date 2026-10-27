import sqlite3
import json
from pathlib import Path

import cv2
import face_recognition
import numpy as np

DB_NAME = "faces.db"

# Smaller tolerance means a stricter match.
# Tune this against your own test images.
MATCH_TOLERANCE = 0.50


def load_database_faces():
    """Load stored face embeddings from the existing database."""
    db_path = Path(DB_NAME)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path.resolve()}")

    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("""
            SELECT face_number, person_label, embedding,
                   embedding_dimension, model_name
            FROM faces
        """).fetchall()

    known_faces = []

    for face_number, label, raw_embedding, dimension, model_name in rows:
        embedding = np.asarray(json.loads(raw_embedding), dtype=np.float64)

        if embedding.ndim != 1 or len(embedding) != dimension:
            print(f"Skipping invalid embedding for face #{face_number}")
            continue

        if not np.all(np.isfinite(embedding)):
            print(f"Skipping non-finite embedding for face #{face_number}")
            continue

        known_faces.append({
            "face_number": face_number,
            "person_label": label,
            "embedding": embedding,
            "model_name": model_name,
        })

    return known_faces


def match_faces(image_path):
    """Detect faces in an image and compare them with database records."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(
        rgb_image,
        number_of_times_to_upsample=1,
        model="hog"
    )

    if not locations:
        print("No faces detected in the input image.")
        return []

    query_embeddings = face_recognition.face_encodings(
        rgb_image,
        known_face_locations=locations,
        num_jitters=1,
        model="small"
    )

    known_faces = load_database_faces()

    if not known_faces:
        print("The database contains no usable face embeddings.")
        return []

    results = []

    for index, (location, query_embedding) in enumerate(
        zip(locations, query_embeddings), start=1
    ):
        query_embedding = np.asarray(query_embedding, dtype=np.float64)
        candidates = []

        for record in known_faces:
            # Do not compare embeddings produced by different models.
            if record["model_name"] != "dlib_face_recognition_resnet":
                continue

            stored_embedding = record["embedding"]

            if stored_embedding.shape != query_embedding.shape:
                continue

            distance = float(np.linalg.norm(
                query_embedding - stored_embedding
            ))

            candidates.append((distance, record))

        top, right, bottom, left = location

        if not candidates:
            result = {
                "input_face": index,
                "match_found": False,
                "reason": "No compatible embeddings in database"
            }
            print(f"Face {index}: no compatible database embeddings.")
            color = (0, 0, 255)
            label_text = "No compatible records"

        else:
            # Smallest Euclidean distance is the closest candidate.
            distance, best_record = min(candidates, key=lambda item: item[0])

            matched = distance <= MATCH_TOLERANCE

            result = {
                "input_face": index,
                "match_found": matched,
                "face_number": (
                    best_record["face_number"] if matched else None
                ),
                "person_label": (
                    best_record["person_label"] if matched else None
                ),
                "distance": round(distance, 4),
                "threshold": MATCH_TOLERANCE
            }

            if matched:
                print(
                    f"Face {index}: MATCH CANDIDATE — "
                    f"record #{best_record['face_number']}, "
                    f"label={best_record['person_label']}, "
                    f"distance={distance:.4f}"
                )
                color = (0, 255, 0)
                label_text = f"Candidate #{best_record['face_number']}"
            else:
                print(
                    f"Face {index}: NO MATCH — "
                    f"closest distance={distance:.4f}"
                )
                color = (0, 0, 255)
                label_text = "No match"

        results.append(result)

        cv2.rectangle(
            image, (left, top), (right, bottom), color, 2
        )
        cv2.putText(
            image,
            label_text,
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    output_path = image_path.with_name(
        f"{image_path.stem}_match_results.jpg"
    )
    cv2.imwrite(str(output_path), image)

    print(f"\nAnnotated image saved to: {output_path.resolve()}")
    return results


if __name__ == "__main__":
    image_file = "faceProfiles/IMG_0258.jpg"

    results = match_faces(image_file)

    print("\nResults:")
    for result in results:
        print(result)
