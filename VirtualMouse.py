import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Streamlit App Title
st.title("AI Virtual Mouse with Hand Gestures")

# Sidebar Settings
st.sidebar.title("Settings")
sensitivity = st.sidebar.slider("Mouse Sensitivity", 1, 10, 5)

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions (virtual for demonstration)
screen_width, screen_height = 1920, 1080


class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Flip for natural interaction

        # Process frame for hand tracking
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get virtual mouse coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)

                # Display the virtual mouse position on screen
                cv2.putText(img, f"Mouse: ({x}, {y})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


# Start webcam streaming using Streamlit WebRTC
webrtc_streamer(key="virtual-mouse", video_transformer_factory=HandGestureTransformer)
