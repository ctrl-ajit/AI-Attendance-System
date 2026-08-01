"""
database.py
Handles all SQLite database operations for the attendance system.
Two tables:
  students   -> id, name, roll_no
  attendance -> id, student_id, date, time
Face encodings are NOT stored in SQLite (they're numpy arrays) -
they're stored separately in encodings.pickle (see enroll.py / recognizer.py)
"""

import sqlite3
from datetime import datetime

DB_PATH = "attendance.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id),
            UNIQUE(student_id, date)
        )
    """)
    conn.commit()
    conn.close()


def add_student(name, roll_no):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO students (name, roll_no) VALUES (?, ?)", (name, roll_no))
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id


def get_all_students():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_attendance(student_id):
    """Marks attendance for today. Ignores if already marked (UNIQUE constraint)."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
            (student_id, today, now_time),
        )
        conn.commit()
        marked = True
    except sqlite3.IntegrityError:
        # already marked today
        marked = False
    conn.close()
    return marked


def get_attendance_records(date=None):
    conn = get_connection()
    if date:
        rows = conn.execute("""
            SELECT s.name, s.roll_no, a.date, a.time
            FROM attendance a JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time
        """, (date,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT s.name, s.roll_no, a.date, a.time
            FROM attendance a JOIN students s ON a.student_id = s.id
            ORDER BY a.date DESC, a.time
        """).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_total_class_days():
    """Total number of distinct dates attendance was ever taken (i.e. number of 'classes' held)."""
    conn = get_connection()
    row = conn.execute("SELECT COUNT(DISTINCT date) as cnt FROM attendance").fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_attendance_summary(defaulter_threshold=75.0):
    """
    Returns per-student attendance stats:
    name, roll_no, days_present, total_class_days, percentage, is_defaulter
    A student is a "defaulter" if their attendance % falls below defaulter_threshold -
    75% is the commonly used academic cutoff.
    """
    total_days = get_total_class_days()
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.id, s.name, s.roll_no, COUNT(a.id) as days_present
        FROM students s
        LEFT JOIN attendance a ON a.student_id = s.id
        GROUP BY s.id
        ORDER BY s.name
    """).fetchall()
    conn.close()

    summary = []
    for row in rows:
        days_present = row["days_present"]
        percentage = (days_present / total_days * 100) if total_days > 0 else 0
        summary.append({
            "name": row["name"],
            "roll_no": row["roll_no"],
            "days_present": days_present,
            "total_class_days": total_days,
            "percentage": round(percentage, 1),
            "is_defaulter": percentage < defaulter_threshold,
        })
    return summary


def get_daily_attendance_counts():
    """Returns [{date, count}] - how many students were marked present each day. Used for the trend chart."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT date, COUNT(*) as count
        FROM attendance
        GROUP BY date
        ORDER BY date
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print("Database initialized: attendance.db")
