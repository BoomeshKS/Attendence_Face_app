# import streamlit as st
# import cv2
# import pandas as pd
# import datetime
# import os
# from PIL import Image
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
# import face_recognition
# import numpy as np

# # File paths
# attendance_file = "attendance/attendance_history.csv"
# user_data_file = "users/user_data.csv"

# # Create necessary directories
# os.makedirs("attendance", exist_ok=True)
# os.makedirs("users", exist_ok=True)

# def load_data(file_path, columns):
#     if os.path.exists(file_path):
#         return pd.read_csv(file_path)
#     else:
#         return pd.DataFrame(columns=columns)

# # Load attendance history and user data
# st.session_state.attendance_history = load_data(attendance_file, ["Name", "Email", "Date", "Time"])
# st.session_state.user_data = load_data(user_data_file, ["Name", "Email", "Image Path", "Date", "Time"])

# if not os.path.exists("registered_faces"):
#     os.makedirs("registered_faces")

# def load_registered_faces():
#     registered_faces = []
#     for root, dirs, files in os.walk("registered_faces"):
#         for file in files:
#             if file.endswith(".jpg") or file.endswith(".png"):
#                 img_path = os.path.join(root, file)
#                 name = file.split("_")[0]
#                 email = file.split("_")[1].split(".")[0]
#                 registered_faces.append({"Name": name, "Email": email, "Image Path": img_path})
#     return registered_faces

# def save_data(file_path, data):
#     data.to_csv(file_path, index=False)

# def clear_all_data():
#     if os.path.exists("attendance"):
#         for file in os.listdir("attendance"):
#             os.remove(os.path.join("attendance", file))
#     if os.path.exists("users"):
#         for file in os.listdir("users"):
#             os.remove(os.path.join("users", file))
#     if os.path.exists("registered_faces"):
#         for file in os.listdir("registered_faces"):
#             os.remove(os.path.join("registered_faces", file))
#     st.session_state.attendance_history = pd.DataFrame(columns=["Name", "Email", "Date", "Time"])
#     st.session_state.user_data = pd.DataFrame(columns=["Name", "Email", "Image Path", "Date", "Time"])

# class FaceRecognitionProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.registered_faces = load_registered_faces()
#         self.known_face_encodings = []
#         self.known_face_names = []
#         for face in self.registered_faces:
#             img = face_recognition.load_image_file(face["Image Path"])
#             encoding = face_recognition.face_encodings(img)[0]
#             self.known_face_encodings.append(encoding)
#             self.known_face_names.append((face["Name"], face["Email"]))

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         rgb_img = img[:, :, ::-1]  # Convert to RGB

#         face_locations = face_recognition.face_locations(rgb_img)
#         face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

#         print(f"Detected {len(face_locations)} faces")  # Debug print

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#             name = "Unknown"
#             email = ""
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name, email = self.known_face_names[first_match_index]
#                 if not ((st.session_state.attendance_history['Name'] == name) & 
#                         (st.session_state.attendance_history['Email'] == email) & 
#                         (st.session_state.attendance_history['Date'] == datetime.date.today().strftime('%Y-%m-%d'))).any():
#                     now = datetime.datetime.now()
#                     new_entry = pd.DataFrame([{"Name": name, "Email": email, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
#                     st.session_state.attendance_history = pd.concat([st.session_state.attendance_history, new_entry], ignore_index=True)
#                     save_data(attendance_file, st.session_state.attendance_history)
            
#             print(f"Drawing box: ({left}, {top}), ({right}, {bottom})")  # Debug print

#             # Draw the rectangle and text
#             cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)  # Thicker rectangle
#             cv2.putText(img, f"Name: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             cv2.putText(img, f"Email: {email}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#         return frame.from_ndarray(img, format="bgr24")

# st.sidebar.title("Face Attendance System")
# menu = st.sidebar.selectbox("Menu", ["Face Attendance", "Register Face", "History", "Clear Data"])

# if menu == "Face Attendance":
#     st.header("Face Attendance")
#     webrtc_streamer(
#         key="attendance",
#         mode=WebRtcMode.SENDRECV,
#         video_processor_factory=FaceRecognitionProcessor,
#         media_stream_constraints={
#             "video": {
#                 "width": {"ideal": 1920},  # Higher resolution width
#                 "height": {"ideal": 1080},  # Higher resolution height
#                 "frameRate": {"ideal": 60},  # Higher frame rate for smoother video
#             },
#             "audio": False,
#         },
#         async_processing=True,
#     )


# elif menu == "Register Face":
#     st.header("Register Face")

#     image_file = st.file_uploader("Upload Image", type=["jpg", "png"], key="register_image")
#     if image_file is not None:
#         img = Image.open(image_file)
#         st.image(img, caption='Uploaded Image', use_column_width=True)
#         name = st.text_input("Name", key="upload_name")
#         email = st.text_input("Email", key="upload_email")
#         if st.button("Save", key="upload_save"):
#             img_path = os.path.join("registered_faces", f"{name}_{email}.jpg")
#             img.save(img_path)
#             now = datetime.datetime.now()
#             new_user = pd.DataFrame([{"Name": name, "Email": email, "Image Path": img_path, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
#             st.session_state.user_data = pd.concat([st.session_state.user_data, new_user], ignore_index=True)
#             save_data(user_data_file, st.session_state.user_data)
#             st.success("Face Registered Successfully and Image Saved")

# elif menu == "History":
#     st.header("Attendance History")
#     st.dataframe(st.session_state.attendance_history)

#     st.header("Registered Users")
#     if not st.session_state.user_data.empty:
#         st.dataframe(st.session_state.user_data[["Name", "Email", "Date", "Time"]])
#     else:
#         st.write("No registered users found.")

#     registered_faces = load_registered_faces()
#     if registered_faces:
#         registered_users_df = pd.DataFrame(registered_faces)
#         st.dataframe(registered_users_df[["Name", "Email"]])
#     else:
#         st.write("No registered users found.")

# elif menu == "Clear Data":
#     st.header("Clear All Data")
#     if st.button("Clear All Data", key="clear_data_button"):
#         clear_all_data()
#         st.success("All data has been cleared.")


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np

st.title("Live Video Face Detection")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="face-detection",
    video_processor_factory=FaceDetectionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    if st.button("Stop"):
        webrtc_ctx.video_processor = None
