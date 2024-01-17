import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from plyer import notification

# Function to send notifications
def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        app_icon=None,
    )

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

# Create empty lists to store face encodings and names
known_face_encodings = []
known_face_names = []

# Specify the folder containing student photos
students_folder = "photos/students/"

# Iterate through files in the folder
for filename in os.listdir(students_folder):
    if filename.lower().endswith((".jpeg", ".jpg", ".png")):
        student_image = face_recognition.load_image_file(os.path.join(students_folder, filename))
        face_encodings = face_recognition.face_encodings(student_image)
        if face_encodings:
            student_encoding = face_encodings[0]
        else:
            print(f"No face found in {filename}. Skipping...")
            continue
        known_face_encodings.append(student_encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use filename without extension as the name

# List to keep track of students
students = list(known_face_names)

# Set up CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file_name = f"attendance_{current_date}.csv"

with open(csv_file_name, 'w+', newline='') as f:
    lnwriter = csv.writer(f)

    # Add headers to the CSV file
    lnwriter.writerow(["Student Name", "Time Present", ""])

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name != "" and name in students and students:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (0, 255, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, f'{name} Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                students.remove(name)
                print(students)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time, ""])

                # Send a notification when a student is marked present
                notification_title = "Attendance Update"
                notification_message = f"{name} marked present at {current_time}"
                send_notification(notification_title, notification_message)

        cv2.imshow("attendance system", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
