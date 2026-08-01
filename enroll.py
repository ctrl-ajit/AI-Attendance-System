"""
enroll.py
Run this once per student to register them into the system.
- Opens the webcam
- Captures a few clear face images
- Computes a face encoding (128-d vector) and averages it
- Saves the encoding into encodings.pickle
- Adds student metadata (name, roll_no) into the SQLite database

Usage:
    python enroll.py
"""

import cv2
import face_recognition
import pickle
import os
from database import init_db, add_student

ENCODINGS_FILE = "encodings.pickle"
NUM_SAMPLES = 5  # number of face captures to average for a stable encoding


def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}  # maps student_id (str) -> encoding (numpy array)


def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def capture_face_encoding():
    """Opens webcam, captures NUM_SAMPLES face encodings, returns their average."""
    cam = cv2.VideoCapture(0)
    encodings_collected = []

    print(f"Look at the camera. Capturing {NUM_SAMPLES} samples... Press 'q' to cancel.")

    while len(encodings_collected) < NUM_SAMPLES:
        ret, frame = cam.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 1:
            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            encodings_collected.append(encoding)
            (top, right, bottom, left) = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {len(encodings_collected)}/{NUM_SAMPLES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(face_locations) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Multiple faces - only one person please", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Enrollment - press q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    if not encodings_collected:
        return None

    # average the samples for a more robust encoding
    import numpy as np
    return np.mean(encodings_collected, axis=0)


def main():
    init_db()
    name = input("Enter student name: ").strip()
    roll_no = input("Enter roll number: ").strip()

    encoding = capture_face_encoding()
    if encoding is None:
        print("Enrollment cancelled - no face captured.")
        return

    student_id = add_student(name, roll_no)

    data = load_encodings()
    data[str(student_id)] = encoding
    save_encodings(data)

    print(f"Enrolled '{name}' (roll no: {roll_no}) successfully with ID {student_id}.")


if __name__ == "__main__":
    main()
