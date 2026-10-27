from pathlib import Path

import cv2
import face_recognition
import numpy as np


# Lower values mean stricter face matching.
MATCH_TOLERANCE = 0.50

# Strong blur effect.
BLUR_KERNEL = (99, 99)


def load_image(image_path):
    """Load an image and return its BGR and RGB versions."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path.resolve()}"
        )

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, rgb_image


def get_face_locations(rgb_image):
    """Detect faces in an image."""
    return face_recognition.face_locations(
        rgb_image,
        number_of_times_to_upsample=1,
        model="hog"
    )


def get_face_embedding(rgb_image, location):
    """Calculate an embedding for one detected face."""
    encodings = face_recognition.face_encodings(
        rgb_image,
        known_face_locations=[location],
        num_jitters=1,
        model="small"
    )

    if not encodings:
        return None

    return np.asarray(encodings[0], dtype=np.float64)


def blur_face(image, location):
    """Blur a face and a margin around it."""
    top, right, bottom, left = location

    height, width = image.shape[:2]

    # Expand the box slightly to cover more of the face area.
    face_height = bottom - top
    face_width = right - left

    margin_y = int(face_height * 0.20)
    margin_x = int(face_width * 0.20)

    top = max(0, top - margin_y)
    bottom = min(height, bottom + margin_y)
    left = max(0, left - margin_x)
    right = min(width, right + margin_x)

    if bottom <= top or right <= left:
        return

    face_region = image[top:bottom, left:right]

    if face_region.size == 0:
        return

    blurred_region = cv2.GaussianBlur(
        face_region,
        BLUR_KERNEL,
        sigmaX=30
    )

    image[top:bottom, left:right] = blurred_region


def process_images(
    reference_image_path,
    group_image_path,
    output_image_path
):
    """
    Use one reference face to identify the same person in a group image.
    Keep matching faces clear and blur all other detected faces.
    """

    # STEP 1: Load the reference image.
    _, reference_rgb = load_image(reference_image_path)

    reference_locations = get_face_locations(reference_rgb)

    if len(reference_locations) != 1:
        raise ValueError(
            "The reference image must contain exactly one detectable face. "
            f"Detected: {len(reference_locations)}"
        )

    reference_embedding = get_face_embedding(
        reference_rgb,
        reference_locations[0]
    )

    if reference_embedding is None:
        raise ValueError(
            "Could not calculate an embedding for the reference face."
        )

    print("Reference face loaded successfully.")

    # STEP 2: Load the group image.
    output_image, group_rgb = load_image(group_image_path)

    group_locations = get_face_locations(group_rgb)

    if not group_locations:
        raise ValueError(
            "No faces were detected in the group image."
        )

    print(f"Detected {len(group_locations)} faces in the group image.")

    # STEP 3: Compare each group face with the reference face.
    results = []

    for index, location in enumerate(group_locations, start=1):
        embedding = get_face_embedding(group_rgb, location)

        # If the face cannot be encoded, blur it to be safe.
        if embedding is None:
            blur_face(output_image, location)

            results.append({
                "face": index,
                "match": False,
                "distance": None,
                "blurred": True
            })

            print(f"Face {index}: embedding failed; blurred.")
            continue

        distance = float(
            np.linalg.norm(reference_embedding - embedding)
        )

        is_match = distance <= MATCH_TOLERANCE

        if is_match:
            print(
                f"Face {index}: MATCH "
                f"(distance={distance:.4f}); left clear."
            )
        else:
            blur_face(output_image, location)

            print(
                f"Face {index}: NO MATCH "
                f"(distance={distance:.4f}); blurred."
            )

        results.append({
            "face": index,
            "match": is_match,
            "distance": round(distance, 4),
            "blurred": not is_match
        })

    # STEP 4: Save the processed group image.
    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), output_image):
        raise IOError(
            f"Could not save output image: {output_path.resolve()}"
        )

    # STEP 5: Print the summary.
    matched_count = sum(item["match"] for item in results)
    blurred_count = sum(item["blurred"] for item in results)

    print("\nProcessing complete.")
    print(f"Total faces detected: {len(results)}")
    print(f"Matching faces left clear: {matched_count}")
    print(f"Faces blurred: {blurred_count}")
    print(f"Output saved to: {output_path.resolve()}")

    return results

