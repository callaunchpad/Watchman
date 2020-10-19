import numpy as np
import cv2

def undistort_image(img, camera = "cam1"):
    if camera == "cam1":
        from data.salsa.calib import cam1 as cam
    elif camera == "cam2":
        from data.salsa.calib import cam2 as cam
    elif camera == "cam3":
        from data.salsa.calib import cam3 as cam
    elif camera == "cam4":
        from data.salsa.calib import cam4 as cam
    
    dst = cv2.undistort(img, cam.intrinsics, cam.distortion)
    return dst