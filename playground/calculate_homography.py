# Required Imports
import numpy as np
import cv2

# Read color image
color_im = cv2.imread('0.png')

# pick source and destination points for perspective warping
pts_src = np.array([[510, 215], [461, 343], [571, 383],[587, 238]])
pts_dst = np.array([[515, 270], [515, 420], [595, 420],[595, 270]])

# draw source points
color_im = cv2.polylines(color_im, [pts_src], True, (0, 255, 0))

# find homography and warp perspective
h, status = cv2.findHomography(pts_src, pts_dst)
im_out = cv2.warpPerspective(color_im, h, (color_im.shape[1], color_im.shape[0]))

cv2.imshow("Source Image", color_im)
cv2.imshow("Destination Image", im_out)
cv2.waitKey(0)
