import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, GUID
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    GUID('{5CDF2C82-841E-4546-9722-0CF74078229A}'),
    CLSCTX_ALL,
    None
)
vc = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = vc.GetVolumeRange()
min_vol, max_vol = volume_range[0], volume_range[1]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

prev_time = 0
volume = 0
volume_bar = 400
volume_percent = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    lm_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([cx, cy])

        if lm_list:
            x1, y1 = lm_list[4]  
            x2, y2 = lm_list[8]   

            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = math.hypot(x2 - x1, y2 - y1)

            volume = np.interp(length, [50, 300], [min_vol, max_vol])
            volume_bar = np.interp(length, [50, 300], [400, 150])
            volume_percent = np.interp(length, [50, 300], [0, 100])

            vc.SetMasterVolumeLevel(volume, None)

            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
            cv2.rectangle(img, (50, int(volume_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volume_percent)} %', (90, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) != 0 else 0
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
