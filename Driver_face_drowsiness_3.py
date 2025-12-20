from collections import deque
import cv2
import dlib
import numpy as np
import pygame
import pyttsx3
import streamlit as st
import os
from scipy.spatial import distance as dist

# --- Paths ---
MODEL_PATH = "/media/darshan/6D5369163EF9BE52/project/shape_predictor_68_face_landmarks.dat"
SOUND_PATH = "/media/darshan/6D5369163EF9BE52/project/mixkit-classic-alarm-995.wav"

# --- Validate files before loading ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Facial landmark model not found at {MODEL_PATH}")
    st.stop()

if not os.path.exists(SOUND_PATH):
    st.error(f"Alert sound file not found at {SOUND_PATH}")
    st.stop()

# --- Streamlit UI ---
st.title("👁️ Eyes Off Road Detector")
start_detection = st.toggle("Start Detection", value=False)

# --- Init ---
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(SOUND_PATH)
engine = pyttsx3.init()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Constants ---
EYE_AR_THRESHOLD = 0.3
CONSEC_FRAMES = 59
EYE_CLOSED_FRAMES = 10
EYE_H_C = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Variables ---
frame_counter = 0
half_drowsy = True
drowsy = True
eye_state_history = deque(maxlen=CONSEC_FRAMES)
half_eye_state_history = deque(maxlen=CONSEC_FRAMES)

# --- Video ---
FRAME_WINDOW = st.image([])

if start_detection:
    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)

    while start_detection:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
            left_eye = shape[42:48]
            right_eye = shape[36:42]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)

            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            # Drowsy logic
            eye_state = 1 if avg_ear < EYE_AR_THRESHOLD else 0
            eye_state_history.append(eye_state)

            if sum(eye_state_history) >= CONSEC_FRAMES - EYE_CLOSED_FRAMES:
                if not drowsy:
                    pygame.mixer.Sound.play(alert_sound)
                    drowsy = True
            else:
                drowsy = False

            # Half-closed logic
            half_eye_state = 1 if avg_ear < EYE_AR_THRESHOLD else 0
            half_eye_state_history.append(half_eye_state)

            if sum(half_eye_state_history) >= CONSEC_FRAMES - EYE_H_C:
                if not half_drowsy:
                    engine.say("You seem drowsy. Please take rest.")
                    engine.runAndWait()
                    half_drowsy = True
            else:
                half_drowsy = False

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Toggle above to start detection.")
