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
from scipy.spatial import distance as dist
from database import init_db, mark_attendance, get_all_students

ENCODINGS_FILE = "encodings.pickle"
TOLERANCE = 0.5  # lower = stricter match. 0.5-0.6 is typical.

# --- Liveness detection (blink check) settings ---
# Prevents marking attendance from a photo held up to the camera -
# a photo can't blink, a real person can.
EAR_THRESHOLD = 0.21      # eye-aspect-ratio below this = eye considered "closed"
EAR_CONSEC_FRAMES = 2     # how many consecutive closed-eye frames count as a real blink


def eye_aspect_ratio(eye_points):
    """
    Computes the Eye Aspect Ratio (EAR) from 6 (x, y) landmark points around one eye.
    EAR drops sharply when the eye closes, and recovers when it opens - so tracking
    this value over time lets us detect a blink.
    Formula from Soukupova & Cech, 2016 ("Real-Time Eye Blink Detection using
    Facial Landmarks").
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = dist.euclidean(p2, p6)
    vertical_2 = dist.euclidean(p3, p5)
    horizontal = dist.euclidean(p1, p4)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


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
    # Tracks blink progress per student across frames:
    # {student_id: {"consec_closed": int, "blinked": bool}}
    blink_tracker = {}
    cam = cv2.VideoCapture(0)

    print("Starting attendance session. Blink to confirm you're a real person. Press 'q' to stop.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        # resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        # landmarks (68-point) give us eye coordinates for blink detection
        face_landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding, landmarks in zip(
            face_locations, face_encodings, face_landmarks_list
        ):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances) if len(distances) else None

            name_label = "Unknown"
            color = (0, 0, 255)

            if best_match_index is not None and distances[best_match_index] < TOLERANCE:
                student_id = ids[best_match_index]
                student = students.get(student_id)
                if student:
                    # --- Liveness check: has this person blinked yet? ---
                    tracker = blink_tracker.setdefault(
                        student_id, {"consec_closed": 0, "blinked": False}
                    )

                    if "left_eye" in landmarks and "right_eye" in landmarks:
                        left_ear = eye_aspect_ratio(landmarks["left_eye"])
                        right_ear = eye_aspect_ratio(landmarks["right_eye"])
                        avg_ear = (left_ear + right_ear) / 2.0

                        if avg_ear < EAR_THRESHOLD:
                            tracker["consec_closed"] += 1
                        else:
                            if tracker["consec_closed"] >= EAR_CONSEC_FRAMES:
                                tracker["blinked"] = True  # eyes closed then reopened = a blink
                            tracker["consec_closed"] = 0

                    if tracker["blinked"]:
                        name_label = f"{student['name']} (verified)"
                        color = (0, 255, 0)

                        if student_id not in already_marked_this_session:
                            marked = mark_attendance(int(student_id))
                            already_marked_this_session.add(student_id)
                            if marked:
                                print(f"Attendance marked: {student['name']} ({student['roll_no']})")
                    else:
                        name_label = f"{student['name']} (blink to verify)"
                        color = (0, 165, 255)  # orange = recognized but not yet verified

            # scale back up face location (since frame was resized by 0.25)
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name_label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Attendance - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()
