import sqlite3
import os
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

DB_PATH = "detections.db"

# ─── DATABASE SETUP ───────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Table for image fake detections
    c.execute("""
        CREATE TABLE IF NOT EXISTS image_detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT NOT NULL,
            result       TEXT NOT NULL,        -- 'FAKE' or 'REAL'
            confidence   REAL NOT NULL,        -- 0.0 to 1.0
            model_used   TEXT DEFAULT 'ResNet-50',
            detected_at  TEXT NOT NULL
        )
    """)

    # Table for voice fake detections
    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT NOT NULL,
            result       TEXT NOT NULL,        -- 'FAKE' or 'REAL'
            confidence   REAL NOT NULL,
            duration_sec REAL DEFAULT 0.0,     -- audio clip duration
            model_used   TEXT DEFAULT 'WaveNet-Detector',
            detected_at  TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # dict-like row access
    return conn


# ─── SEED SAMPLE DATA (for demo) ─────────────────────────────────
def seed_demo_data():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM image_detections")
    if c.fetchone()[0] > 0:
        conn.close()
        return  # already seeded

    image_files = ["face_001.jpg","photo_scan.png","user_upload.jpg",
                   "deepfake_test.jpg","real_photo.jpg","synthetic_gen.png",
                   "portrait_ai.jpg","news_img.jpg","social_post.jpg","id_scan.jpg"]

    voice_files = ["audio_clip.wav","voice_msg.mp3","call_record.wav",
                   "interview.mp3","synthetic_voice.wav","news_audio.wav",
                   "deepfake_audio.mp3","real_speech.wav","podcast_seg.mp3","phone_call.wav"]

    base = datetime.now() - timedelta(days=7)
    for i, f in enumerate(image_files):
        result = "FAKE" if i % 3 != 0 else "REAL"
        conf = round(random.uniform(0.72, 0.99), 2)
        ts = (base + timedelta(hours=i*8)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO image_detections (filename,result,confidence,detected_at) VALUES (?,?,?,?)",
                  (f, result, conf, ts))

    for i, f in enumerate(voice_files):
        result = "FAKE" if i % 4 != 0 else "REAL"
        conf = round(random.uniform(0.68, 0.98), 2)
        dur = round(random.uniform(2.0, 45.0), 1)
        ts = (base + timedelta(hours=i*9+2)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO voice_detections (filename,result,confidence,duration_sec,detected_at) VALUES (?,?,?,?,?)",
                  (f, result, conf, dur, ts))

    conn.commit()
    conn.close()


# ─── IMAGE DETECTION ROUTES ───────────────────────────────────────
@app.route("/api/image/detect", methods=["POST"])
def image_detect():
    """Simulate detection and save to DB"""
    data = request.get_json()
    filename = data.get("filename", "unknown.jpg")
    result = random.choice(["FAKE", "FAKE", "REAL"])       # biased demo
    confidence = round(random.uniform(0.75, 0.99), 2)
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db()
    conn.execute(
        "INSERT INTO image_detections (filename,result,confidence,detected_at) VALUES (?,?,?,?)",
        (filename, result, confidence, detected_at)
    )
    conn.commit()
    conn.close()

    return jsonify({"result": result, "confidence": confidence,
                    "filename": filename, "detected_at": detected_at}), 201


@app.route("/api/image/history", methods=["GET"])
def image_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM image_detections ORDER BY detected_at DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ─── VOICE DETECTION ROUTES ───────────────────────────────────────
@app.route("/api/voice/detect", methods=["POST"])
def voice_detect():
    data = request.get_json()
    filename = data.get("filename", "unknown.wav")
    duration = round(random.uniform(3.0, 30.0), 1)
    result = random.choice(["FAKE", "FAKE", "REAL"])
    confidence = round(random.uniform(0.70, 0.97), 2)
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db()
    conn.execute(
        "INSERT INTO voice_detections (filename,result,confidence,duration_sec,detected_at) VALUES (?,?,?,?,?)",
        (filename, result, confidence, duration, detected_at)
    )
    conn.commit()
    conn.close()

    return jsonify({"result": result, "confidence": confidence,
                    "filename": filename, "duration_sec": duration,
                    "detected_at": detected_at}), 201


@app.route("/api/voice/history", methods=["GET"])
def voice_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM voice_detections ORDER BY detected_at DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ─── ANALYTICS ROUTE ──────────────────────────────────────────────
@app.route("/api/analytics", methods=["GET"])
def analytics():
    conn = get_db()

    def stats(table):
        total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        fake  = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE result='FAKE'").fetchone()[0]
        real  = total - fake
        avg_conf = conn.execute(f"SELECT AVG(confidence) FROM {table}").fetchone()[0]
        return {
            "total": total, "fake": fake, "real": real,
            "fake_pct": round(fake / total * 100, 1) if total else 0,
            "avg_confidence": round(avg_conf * 100, 1) if avg_conf else 0
        }

    # Daily trend (last 7 days) — image
    daily_img = conn.execute("""
        SELECT substr(detected_at,1,10) as day, result, COUNT(*) as cnt
        FROM image_detections
        WHERE detected_at >= date('now','-7 days')
        GROUP BY day, result ORDER BY day
    """).fetchall()

    daily_voice = conn.execute("""
        SELECT substr(detected_at,1,10) as day, result, COUNT(*) as cnt
        FROM voice_detections
        WHERE detected_at >= date('now','-7 days')
        GROUP BY day, result ORDER BY day
    """).fetchall()

    conn.close()
    return jsonify({
        "image": stats("image_detections"),
        "voice": stats("voice_detections"),
        "daily_image": [dict(r) for r in daily_img],
        "daily_voice": [dict(r) for r in daily_voice]
    })


# ─── SERVE FRONTEND ───────────────────────────────────────────────
@app.route("/layer1")
def layer1():
    return send_from_directory("static", "layer1.html")


if __name__ == "__main__":
    init_db()
    seed_demo_data()
    print("DeepDetect running → http://localhost:9000")
    app.run(debug=True)
