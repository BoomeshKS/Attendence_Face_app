
import cv2
import streamlit as st
import numpy as np
import os
from datetime import datetime
import pandas as pd
from streamlit_option_menu import option_menu

# Function to detect faces using Haar cascades
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, gray

# Function to load registered faces and their encodings
def load_registered_faces():
    registered_faces = []
    labels = []
    names = {}
    label_id = 0
    for filename in os.listdir('Registered_face'):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join('Registered_face', filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            name = filename.split('_')[0]
            registered_faces.append(image)
            labels.append(label_id)
            names[label_id] = name
            label_id += 1
    return registered_faces, labels, names

# Function to train the face recognizer
def train_recognizer(registered_faces, labels):
    if len(registered_faces) == 0 or len(labels) == 0:
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(registered_faces, np.array(labels))
    return recognizer

# Function to load attendance history
def load_attendance_history(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame(columns=["Name", "Email", "Timestamp"])
    return pd.read_csv(file_path)

# Function to save attendance history
def save_attendance_history(file_path, history_df):
    history_df.to_csv(file_path, index=False)

# Ensure the "Registered_face" directory exists
if not os.path.exists('Registered_face'):
    os.makedirs('Registered_face')

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load registered faces and train the recognizer
registered_faces, labels, names = load_registered_faces()
recognizer = train_recognizer(registered_faces, labels)

# Load attendance history
attendance_history_file = "attendance_history.csv"
attendance_history = load_attendance_history(attendance_history_file)

# Streamlit application
st.title("Face Recognition System")

# Create the menu options
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Live Attendance", "Register Face", "Register Face Using Camera", "History", "Clear Data"],
        icons=["camera-video", "person-plus", "camera", "clock-history", "trash"],
        menu_icon="cast",
        default_index=0,
    )

# Define the live attendance functionality
if selected == "Live Attendance":
    st.header("Live Attendance")

    if recognizer is None:
        st.error("No registered faces found. Please register at least one face before starting the recognizer.")
    else:
        start_button = st.button("Start", key="start_button")

        if start_button:
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            stop_button = st.button("Stop", key="stop_button")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break

                faces, gray = detect_faces(frame, face_cascade)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    label_id, confidence = recognizer.predict(roi_gray)

                    if confidence < 50:
                        name = "Unknown"
                        email = "new face"
                        color = (255, 0, 0)  # Blue for new faces
                    else:
                        name = names[label_id]
                        # Fetch email from the history.csv
                        with open("history.csv", 'r') as file:
                            lines = file.readlines()
                            email = ""
                            for line in lines:
                                parts = line.strip().split(',')
                                if parts[0] == name:
                                    email = parts[1]
                                    break
                        color = (0, 0, 255)  # Red for recognized faces

                        # Add entry to attendance history if not already present
                        if not ((attendance_history["Name"] == name) & (attendance_history["Email"] == email)).any():
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            new_entry = pd.DataFrame([{"Name": name, "Email": email, "Timestamp": timestamp}])
                            attendance_history = pd.concat([attendance_history, new_entry], ignore_index=True)
                            save_attendance_history(attendance_history_file, attendance_history)

                    display_text = f"{name}, {email}"
                    
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    # Draw label with name below the face
                    cv2.rectangle(frame, (x, y + h + 10), (x + w, y + h + 40), color, cv2.FILLED)
                    cv2.putText(frame, display_text, (x + 6, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                frame_placeholder.image(frame, channels="BGR")

            cap.release()
            frame_placeholder.empty()

# Define the register face functionality
elif selected == "Register Face":
    st.header("Register Face")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        st.image(image, channels="BGR")
        with st.form(key='upload_form'):
            name = st.text_input("Name", key="upload_name")
            email = st.text_input("Email", key="upload_email")
            save_button = st.form_submit_button("Save")
            if save_button and name and email:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Registered_face/{name}_{timestamp}.jpg"
                try:
                    cv2.imwrite(filename, image)
                    with open("history.csv", "a") as f:
                        f.write(f"{name},{email},{timestamp}\n")
                    st.success(f"Image saved successfully as {filename}")
                except Exception as e:
                    st.error(f"Failed to save image: {e}")

                # Refresh registered faces and retrain recognizer
                registered_faces, labels, names = load_registered_faces()
                recognizer = train_recognizer(registered_faces, labels)
                st.experimental_rerun()

# Define the register face using camera functionality
elif selected == "Register Face Using Camera":
    st.header("Register Face Using Camera")
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    capture_button = st.button("Capture Image")

    while cap.isOpened() and st.session_state.captured_image is None:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            cap.release()
            break

        faces, gray = detect_faces(frame, face_cascade)
        frame_placeholder.image(frame, channels="BGR")

        if capture_button:
            st.session_state.captured_image = frame.copy()
            cap.release()
            break

    if st.session_state.captured_image is not None:
        st.image(st.session_state.captured_image, channels="BGR")
        with st.form(key='camera_form'):
            name = st.text_input("Name", key="camera_name")
            email = st.text_input("Email", key="camera_email")
            save_button = st.form_submit_button("Save")
            if save_button and name and email:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Registered_face/{name}_{timestamp}.jpg"
                try:
                    cv2.imwrite(filename, st.session_state.captured_image)
                    with open("history.csv", "a") as f:
                        f.write(f"{name},{email},{timestamp}\n")
                    st.success(f"Image saved successfully as {filename}")
                except Exception as e:
                    st.error(f"Failed to save image: {e}")

                # Refresh registered faces and retrain recognizer
                registered_faces, labels, names = load_registered_faces()
                recognizer = train_recognizer(registered_faces, labels)
                st.session_state.captured_image = None
                st.experimental_rerun()

# Define the history functionality
elif selected == "History":
    # Function to load CSV data with validation
    def load_csv(file_path):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            valid_lines = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 3:  # Expecting Name, Email, Timestamp
                    valid_lines.append(line.strip())
                else:
                    st.error(f"Invalid line in {file_path}: {line}")
            if valid_lines:
                data = pd.DataFrame([line.split(',') for line in valid_lines], columns=["Name", "Email", "Timestamp"])
                return data, None
            else:
                return None, "No valid data found in the file."
        except Exception as e:
            return None, f"Error reading the file: {e}"

    # Registered Face History Table
    st.subheader("Registered Face History")
    registered_face_history, error = load_csv("history.csv")
    if registered_face_history is not None:
        st.table(registered_face_history)
    else:
        st.info(error)

    # Attendance History Table
    st.subheader("Attendance History")
    attendance_history, error = load_csv(attendance_history_file)
    if attendance_history is not None:
        st.table(attendance_history)
    else:
        st.info(error)

# Define the clear data functionality
elif selected == "Clear Data":
    st.header("Clear Data")
    clear_data_button = st.button("Clear All Data")

    if clear_data_button:
        # Clear images in Registered_face directory
        for filename in os.listdir('Registered_face'):
            file_path = os.path.join('Registered_face', filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clear history.csv
        with open("history.csv", 'w') as f:
            f.write("")

        # Clear attendance_history.csv
        with open(attendance_history_file, 'w') as f:
            f.write("")

        st.success("All data cleared successfully.")

