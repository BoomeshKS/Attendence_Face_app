from flask import Flask, Response, g
import cv2
import face_recognition
import sqlite3
import numpy as np

app = Flask(__name__)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('faces.db')
        create_table(db)
    return db

def create_table(db):
    c = db.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    ''')
    db.commit()

# Load known faces from database
def load_known_faces():
    with app.app_context():
        conn = get_db()
        c = conn.cursor()
        known_face_encodings = []
        known_face_names = []
        for row in c.execute('SELECT name, encoding FROM faces'):
            known_face_names.append(row[0])
            known_face_encodings.append(np.frombuffer(row[1], dtype=np.float64))
    return known_face_names, known_face_encodings

known_face_names, known_face_encodings = load_known_faces()

def gen_frames():
    video_capture = cv2.VideoCapture(0)
    face_encoding = None
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        match_found = False
        for face_encoding in face_encodings:
            with app.app_context():
                conn = get_db()
                c = conn.cursor()
                existing_encoding = c.execute('SELECT 1 FROM faces WHERE encoding = ?', (sqlite3.Binary(face_encoding.tobytes()),)).fetchone()
            if existing_encoding:
                match_found = True
                break

        if not match_found:
            if face_encoding is not None:
                unknown_encoding = sqlite3.Binary(face_encoding.tobytes())
                with app.app_context():
                    conn = get_db()
                    c = conn.cursor()
                    c.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', ('Unknown', unknown_encoding))
                    conn.commit()
                    print("New face found. Encoding saved to database.")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition</title>
    </head>
    <body>
        <h1>Face Recognition</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
