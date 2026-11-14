"""
Enroll a person into the IBVAP known-faces watchlist.

Usage:
    python enroll_face.py --name "John Doe" --image path/to/photo.jpg
    python enroll_face.py --name "John Doe" --webcam
"""

import argparse
import os
import shutil

import cv2

from face_id import FaceIdentifier


def enroll_from_image(name, image_path, known_faces_dir="known_faces"):
    person_dir = os.path.join(known_faces_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    dest = os.path.join(person_dir, os.path.basename(image_path))
    shutil.copy(image_path, dest)

    print(f"Copied {image_path} -> {dest}")


def enroll_from_webcam(name, known_faces_dir="known_faces", num_shots=5):
    person_dir = os.path.join(known_faces_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture a photo, ESC to finish early.")

    count = 0

    while count < num_shots:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Enroll - press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            count += 1
            path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved {path} ({count}/{num_shots})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll a face into the IBVAP watchlist"
    )
    parser.add_argument("--name", required=True, help="Person's name / identity label")
    parser.add_argument("--image", help="Path to a photo of the person")
    parser.add_argument(
        "--webcam", action="store_true", help="Capture photos from webcam instead"
    )
    parser.add_argument(
        "--shots", type=int, default=5, help="Number of webcam shots to capture"
    )

    args = parser.parse_args()

    if args.webcam:
        enroll_from_webcam(args.name, num_shots=args.shots)
    elif args.image:
        enroll_from_image(args.name, args.image)
    else:
        parser.error("Provide either --image or --webcam")

    print("Rebuilding face encodings...")
    identifier = FaceIdentifier()
    total = identifier.build_encodings()
    print(f"Done. {total} face(s) now in the watchlist.")
