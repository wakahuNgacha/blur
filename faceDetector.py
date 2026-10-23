import cv2

def detect_faces(image_path):
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected.")
    else:
        print(f"Detected {len(faces)} face(s).")
        for (x, y, w, h) in faces:
            # Draw a rectangle around each detected face
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the output image with detected faces
        cv2.imshow('Detected Faces', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # return faces

detect_faces("faceProfiles/WhatsApp Image 2026-10-23 at 18.39.04.jpeg")
