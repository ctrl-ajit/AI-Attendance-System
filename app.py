"""
app.py
Flask web dashboard for the attendance system.

Note: Enrollment and live recognition use OpenCV windows (cv2.imshow),
which need direct webcam access - run those as standalone scripts:
    python enroll.py
    python recognizer.py

This Flask app is the "management" layer: view students, view attendance,
export records as CSV. Run with:
    python app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, Response
import pandas as pd
import json
from database import (
    init_db, get_all_students, get_attendance_records,
    get_attendance_summary, get_daily_attendance_counts,
)

app = Flask(__name__)
init_db()


@app.route("/")
def home():
    students = get_all_students()
    return render_template("index.html", students=students)


@app.route("/attendance")
def attendance():
    date_filter = request.args.get("date")  # e.g. 2026-08-01, optional
    records = get_attendance_records(date_filter)
    return render_template("attendance.html", records=records, date_filter=date_filter)


@app.route("/analytics")
def analytics():
    summary = get_attendance_summary(defaulter_threshold=75.0)
    daily_counts = get_daily_attendance_counts()

    chart_labels = json.dumps([d["date"] for d in daily_counts])
    chart_values = json.dumps([d["count"] for d in daily_counts])

    return render_template(
        "analytics.html",
        summary=summary,
        chart_labels=chart_labels,
        chart_values=chart_values,
    )


@app.route("/export")
def export_csv():
    date_filter = request.args.get("date")
    records = get_attendance_records(date_filter)
    df = pd.DataFrame(records)
    csv_data = df.to_csv(index=False)
    filename = f"attendance_{date_filter or 'all'}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"},
    )


if __name__ == "__main__":
    app.run(debug=True)
