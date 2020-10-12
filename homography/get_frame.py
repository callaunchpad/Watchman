import cv2
import os
from glob import glob
import numpy as np

vid_num = 1
cap = cv2.VideoCapture(f"video/salsa_ps_cam{vid_num}.avi")
total_frames = cap.get(7)

n = 2000
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if i == n:
        cv2.imwrite(f"frame{vid_num}.jpg", frame)
        break
    i += 1
    #cv2.imshow("frame2", frame)
    #cv2.waitKey(0)

print(total_frames)