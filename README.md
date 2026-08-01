# AI Attendance System using Facial Recognition

Final year project starter - IT / Python / AI-ML.

## How it works
1. `enroll.py` - captures a student's face via webcam, saves a face "encoding"
   (a 128-number vector, not the raw image) into `encodings.pickle`, and stores
   name/roll number in `attendance.db` (SQLite).
2. `recognizer.py` - opens the webcam, detects faces frame by frame, compares
   each detected face to all stored encodings, and if it finds a confident
   match, marks that student present (once per day) in the database.
3. `app.py` - a Flask web dashboard to view enrolled students, view/filter
   attendance records, and export them as CSV.

## Setup

### 1. Install Python 3.9+ 
Check with `python --version`.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Important note on `face_recognition`:** it depends on `dlib`, which needs a
C++ compiler to build.
- **Windows:** install "Desktop development with C++" via Visual Studio
  Build Tools first, OR install dlib via conda instead:
  `conda install -c conda-forge dlib` then `pip install face_recognition`.
- **Mac:** `brew install cmake` first, then pip install should work.
- **Linux:** `sudo apt install cmake` first, then pip install should work.

If you get stuck here, this is the #1 most common blocker for this kind of
project - budget time for it early, don't leave it for the night before.

### 3. Run enrollment (do this once per student)
```bash
python enroll.py
```
Follow the on-screen prompts and let it capture your face samples.
Repeat for each student you want in the system.

### 4. Run live attendance
```bash
python recognizer.py
```
Opens your webcam. Recognized faces get a green box + name and are marked
present automatically. Press `q` to stop the session.

### 5. View the dashboard
```bash
python app.py
```
Open http://127.0.0.1:5000 in your browser to see enrolled students and
attendance records, and to export CSVs.

## Project structure
```
attendance_system/
├── app.py              # Flask dashboard
├── database.py         # SQLite schema + helper functions
├── enroll.py            # Webcam enrollment script
├── recognizer.py         # Webcam live recognition + attendance marking
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html
│   └── attendance.html
├── attendance.db        # created automatically on first run
└── encodings.pickle     # created automatically after first enrollment
```

## Ideas to extend (if you finish early / want to boost your report)
- **Liveness detection** - blink detection or asking the user to turn their
  head, to prevent someone holding up a photo instead of a real face.
- **Email alerts** - notify a mentor/parent if attendance drops below a
  threshold (use `smtplib`).
- **Analytics page** - attendance % per student over time, chart with
  matplotlib or Chart.js.
- **Multiple camera / classroom mode** - recognize multiple students in one
  frame at once (the code already supports this - `recognizer.py` loops over
  all detected faces per frame).

## For your report
Suggested sections: Abstract, Introduction & Problem Statement, Literature
Survey (cite a couple of face recognition papers - FaceNet, dlib's HOG+SVM
approach), System Architecture (diagram: webcam -> face detection -> encoding
-> matching -> database -> dashboard), Implementation, Results/Screenshots,
Limitations (lighting, occlusion, similar-looking faces), Future Scope,
Conclusion.
