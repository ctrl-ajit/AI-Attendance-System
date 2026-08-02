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

# Set to True temporarily to print live edge_density / skin_ratio values to the
# terminal - use this to figure out the right threshold values for your own
# webcam and lighting, then set back to False once tuned.
DEBUG_ACCESSORY_VALUES = False

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


# --- Mask detection settings ---
# Lightweight heuristic (no separate trained model needed): looks at the lower
# half of the face box (nose/mouth region) and measures edge density using
# Canny edge detection. A bare mouth/nose area has a lot of fine detail (lips,
# nostrils, teeth, shadows under the nose) which produces many edges. A cloth
# or surgical mask is comparatively flat and uniform, producing far fewer
# edges. This is a simple proxy, not a trained classifier - a CNN-based mask
# classifier would be more robust and is noted as future scope.
MASK_EDGE_DENSITY_THRESHOLD = 0.065


def get_lower_face_edge_density(frame, top, right, bottom, left):
    """Returns the raw edge-density value for the lower-face region (0.0 to 1.0)."""
    face_height = bottom - top
    lower_top = top + face_height // 2
    lower_region = frame[lower_top:bottom, left:right]

    if lower_region.size == 0:
        return 1.0  # treat empty region as "not masked" (fail safe)

    gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.count_nonzero(edges) / edges.size


# --- Cap/hat detection settings ---
# Another lightweight heuristic: a cap covers the forehead (the region between
# the top of the face box and the eyebrows). We check what fraction of that
# region looks like skin tone using an HSV color threshold. Bare forehead =
# mostly skin-colored pixels. A cap (usually fabric, any color) = low skin
# ratio in that region.
# Known limitation (worth stating honestly in the report): fixed HSV skin
# thresholds work less reliably across all skin tones and lighting conditions -
# a trained classifier would be more robust. Same caveat applies to hair
# covering the forehead, which can also trigger a false "cap detected".
CAP_SKIN_RATIO_THRESHOLD = 0.15
SKIN_HSV_LOWER = np.array([0, 20, 70], dtype=np.uint8)
SKIN_HSV_UPPER = np.array([20, 150, 255], dtype=np.uint8)


def get_forehead_skin_ratio(frame, top, right, left, landmarks, scale_factor=4):
    """Returns the raw skin-tone ratio (0.0 to 1.0) for the forehead region. Returns None if it can't be measured."""
    if "left_eyebrow" not in landmarks or "right_eyebrow" not in landmarks:
        return None

    eyebrow_points = landmarks["left_eyebrow"] + landmarks["right_eyebrow"]
    eyebrow_ys = [p[1] * scale_factor for p in eyebrow_points]
    forehead_bottom = min(eyebrow_ys)

    if forehead_bottom <= top:
        return None

    forehead_region = frame[top:forehead_bottom, left:right]
    if forehead_region.size == 0:
        return None

    hsv = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    return np.count_nonzero(skin_mask) / skin_mask.size


def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print("No encodings found. Run enroll.py first.")
        return {}
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


# --- Lighting sufficiency check ---
# Both the mask and cap heuristics rely on seeing real detail/color in the
# face - which fails under backlighting or a dark room, producing false
# "mask"/"cap detected" results even on a bare face. Rather than reporting
# an unreliable guess, we check overall brightness first and skip the
# accessory checks entirely if it's too dark, showing a lighting warning
# instead. This is a known, documented limitation of heuristic (non-ML)
# accessory detection.
MIN_BRIGHTNESS_FOR_ACCESSORY_CHECK = 60  # 0-255 grayscale scale


def get_face_brightness(frame, top, right, bottom, left):
    face_region = frame[top:bottom, left:right]
    if face_region.size == 0:
        return 255  # assume bright enough if we can't measure, fail safe
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


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
            # scale up face location now (frame was resized by 0.25 for detection speed)
            # so both mask-detection and drawing use full-resolution pixels
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            brightness = get_face_brightness(frame, top, right, bottom, left)
            lighting_ok = brightness >= MIN_BRIGHTNESS_FOR_ACCESSORY_CHECK

            if lighting_ok:
                edge_density = get_lower_face_edge_density(frame, top, right, bottom, left)
                skin_ratio = get_forehead_skin_ratio(frame, top, right, left, landmarks)
                mask_detected = edge_density < MASK_EDGE_DENSITY_THRESHOLD
                cap_detected = skin_ratio is not None and skin_ratio < CAP_SKIN_RATIO_THRESHOLD
            else:
                edge_density, skin_ratio = None, None
                mask_detected, cap_detected = False, False

            if DEBUG_ACCESSORY_VALUES:
                edge_display = f"{edge_density:.3f}" if edge_density is not None else "N/A"
                skin_display = f"{skin_ratio:.3f}" if skin_ratio is not None else "N/A"
                print(f"[debug] brightness={brightness:.1f} (need >= {MIN_BRIGHTNESS_FOR_ACCESSORY_CHECK})  "
                      f"edge_density={edge_display}  skin_ratio={skin_display}  lighting_ok={lighting_ok}")

            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances) if len(distances) else None

            name_label = "Unknown"
            color = (0, 0, 255)

            if best_match_index is not None and distances[best_match_index] < TOLERANCE:
                student_id = ids[best_match_index]
                student = students.get(student_id)
                if student:
                    # --- Liveness check: has this person blinked yet? ---
                    # Eyes stay visible even with a mask on, so this still works.
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

                    tags = []
                    if mask_detected:
                        tags.append("MASK")
                    if cap_detected:
                        tags.append("CAP")
                    if not lighting_ok:
                        tags.append("LOW LIGHT")
                    accessory_tag = f" [{', '.join(tags)}]" if tags else ""

                    if tracker["blinked"]:
                        name_label = f"{student['name']} (verified){accessory_tag}"
                        color = (0, 255, 0)

                        if student_id not in already_marked_this_session:
                            marked = mark_attendance(int(student_id))
                            already_marked_this_session.add(student_id)
                            if marked:
                                tag_note = f" [{', '.join(tags)}]" if tags else ""
                                print(f"Attendance marked: {student['name']} ({student['roll_no']}){tag_note}")
                    else:
                        name_label = f"{student['name']} (blink to verify){accessory_tag}"
                        color = (0, 165, 255)  # orange = recognized but not yet verified

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name_label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Attendance - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_recognition()
