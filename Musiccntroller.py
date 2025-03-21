import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pygame
import pygame.camera
import mediapipe as mp
import numpy as np
import cv2
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Spotify authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-modify-playback-state user-read-playback-state"
))

pygame.init()

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Pygame camera
pygame.camera.init()
cameras = pygame.camera.list_cameras()
if not cameras:
    raise Exception("No camera found!")
camera = pygame.camera.Camera(cameras[0], (640, 480))
camera.start()

# Create Pygame window
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Spotify Gesture Control")

LEFT_HAND_COLOR = (255, 0, 0)  # Red for volume control
RIGHT_HAND_COLOR = (0, 255, 0)  # Green for track control

# Store the time of the last volume adjustment
last_volume_adjustment = 0
volume_delay = 0.5  # delay in seconds

def safe_spotify_request(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            print("Rate limit reached. Retrying...")
            time.sleep(2)  # Shorter sleep time to avoid long delays
            return safe_spotify_request(func, *args, **kwargs)
        else:
            raise
    return None

running = True
while running:
    frame = camera.get_image()
    frame_surface = pygame.surfarray.array3d(frame)
    frame_surface = np.rot90(frame_surface)
    frame_surface = cv2.flip(frame_surface, 0)
    
    rgb_frame = cv2.cvtColor(frame_surface, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_x = hand_landmarks.landmark[0].x
            left_hand = hand_x < 0.5
            right_hand = hand_x >= 0.5
            
            color = LEFT_HAND_COLOR if left_hand else RIGHT_HAND_COLOR
            mp_drawing.draw_landmarks(
                frame_surface,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2)
            )
            
            # Volume control (left hand - thumb to index finger distance)
            if left_hand:
                index_finger = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]
                distance_left = np.linalg.norm(
                    np.array([index_finger.x, index_finger.y]) - np.array([thumb.x, thumb.y])
                )
                volume = int(np.interp(distance_left, [0.05, 0.35], [0, 100]))

                # Check if enough time has passed since the last volume change
                current_time = time.time()
                if current_time - last_volume_adjustment > volume_delay:
                    if safe_spotify_request(sp.volume, volume):
                        print(f"Volume set to: {volume}%")
                    last_volume_adjustment = current_time

            # Track control (right hand - thumb to index finger tapping)
            if right_hand:
                index_finger = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]
                distance_right = np.linalg.norm(
                    np.array([index_finger.x, index_finger.y]) - np.array([thumb.x, thumb.y])
                )
                
                if distance_right < 0.05:
                    if safe_spotify_request(sp.next_track):
                        print("Skipped to next track")

    frame_surface = np.rot90(frame_surface)
    screen.blit(pygame.surfarray.make_surface(frame_surface), (0, 0))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

camera.stop()
pygame.quit()
 