"""
recognizer.py
Runs live face recognition from the webcam and marks attendance
for recognized students (once per day, per student).

Usage:
    python recognizer.py
Press 'q' to stop.
"""

import cv2
import face_recognition
import pickle
import os
import numpy as np
from database import init_db, mark_attendance, get_all_students

ENCODINGS_FILE = "encodings.pickle"
TOLERANCE = 0.5  # lower = stricter match. 0.5-0.6 is typical.


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print("No encodings found. Run enroll.py first.")
        return {}
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def build_lookup():
    """Returns (list_of_ids, list_of_encodings, id_to_name_map)."""
    encodings_data = load_encodings()
    students = {str(s["id"]): s for s in get_all_students()}

    ids, encodings = [], []
    for student_id, encoding in encodings_data.items():
        ids.append(student_id)
        encodings.append(encoding)

    return ids, encodings, students


def run_recognition():
    init_db()
    ids, known_encodings, students = build_lookup()

    if not known_encodings:
        return

    already_marked_this_session = set()
    cam = cv2.VideoCapture(0)

    print("Starting attendance session. Press 'q' to stop.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances) if len(distances) else None

            name_label = "Unknown"
            color = (0, 0, 255)

            if best_match_index is not None and distances[best_match_index] < TOLERANCE:
                student_id = ids[best_match_index]
                student = students.get(student_id)
                if student:
                    name_label = student["name"]
                    color = (0, 255, 0)

                    if student_id not in already_marked_this_session:
                        marked = mark_attendance(int(student_id))
                        already_marked_this_session.add(student_id)
                        if marked:
                            print(f"Attendance marked: {student['name']} ({student['roll_no']})")

            # scale back up face location (since frame was resized by 0.25)
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name_label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()
