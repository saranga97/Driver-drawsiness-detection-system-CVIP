import tkinter as tk
from tkinter import ttk
import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
from pygame import mixer

# Initialize mixer for playing alert sound
mixer.init()
mixer.music.load("music.wav")


# Eye aspect ratio calculation function
def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


# Constants for drowsiness detection
threshold = 0.25
frame_check = 20
flag = 0

closed_eye_frames_threshold = 0.3 * 20  # 4 seconds assuming 20 frames per second

closed_eye_frames_count = 0
drowsy_detected = False

# Initialize variables for evaluation
correct_drowsy_predictions = 0
incorrect_drowsy_predictions = 0

# Eye landmarks of the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

# Initialize face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to start drowsiness detection
def start_detection():
    global closed_eye_frames_count, drowsy_detected

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=650)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape_gray = predict(gray, subject)
            shape = face_utils.shape_to_np(shape_gray)

            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            if ear < threshold:
                closed_eye_frames_count += 1
                if closed_eye_frames_count >= closed_eye_frames_threshold:
                    if not drowsy_detected:
                        print("**** ALERT ****")
                        mixer.music.play()
                        drowsy_detected = True
            else:
                closed_eye_frames_count = 0
                drowsy_detected = False

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


# Function to stop drowsiness detection
def stop_detection():
    root.destroy()
    cap = cv2.VideoCapture(0)
    cv2.destroyAllWindows()
    cap.release()

# Create the Tkinter application window
root = tk.Tk()
root.title("Drowsiness Detection System")

# Set window size
root.geometry("400x300")

# Create style for buttons
style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12), background="#4CAF50")  # Set default background color
style.map("TButton", background=[("active", "#45a049")])  # Set hover color

# Create label mentioning the application's origin
label = tk.Label(root, text="Application by Faculty of Engineering, University of Ruhuna", font=("Arial", 10))
label.pack(side="top", pady=5)

# Create buttons for starting and stopping detection
# Using grid for better control over button placement
frame = tk.Frame(root)  # Create a frame to hold the buttons
frame.pack(expand=True)  # Allow frame to fill available space

start_button = ttk.Button(frame, text="Start Detection", command=start_detection, padding=10)
start_button.grid(row=0, column=0, pady=(10, 20), sticky="nsew")  # Add vertical margins (top, bottom) with "nsew" for anchor

stop_button = ttk.Button(frame, text="Stop Detection", command=stop_detection, padding=10)
stop_button.grid(row=1, column=0, pady=(0, 10), sticky="nsew")  # Adjust margins for spacing

# Run the Tkinter event loop
root.mainloop()

