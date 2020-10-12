import cv2
import numpy as np

vidcap1 = cv2.VideoCapture('../../salsa_ps_cam1.avi')
vidcap2 = cv2.VideoCapture('../../salsa_ps_cam2.avi')
success,image1_raw = vidcap1.read()
success,image2_raw = vidcap2.read()

pts_src1 = np.array([[817, 345], [733, 551], [913, 611],[939, 384]])
pts_dst1 = np.array([[870, 400], [870, 650], [1000, 650],[1000, 400]])

pts_src2 = np.array([[411, 604], [532, 609], [530, 720],[393, 712]])
pts_dst2 = np.array([[100, 350], [100, 300], [50, 300], [50, 350]])

image1 = cv2.polylines(image1_raw, [pts_src1], True, (0, 255, 0))
image2 = cv2.polylines(image2_raw, [pts_src2], True, (0, 255, 0))

h1, status = cv2.findHomography(pts_src1, pts_dst1)
im_out1 = cv2.warpPerspective(image1, h1, (image1.shape[1], image1.shape[0]))

h2, status = cv2.findHomography(pts_src2, pts_dst2)
im_out2 = cv2.warpPerspective(image2, h2, (image2.shape[1], image2.shape[0]))

# count = 0
# while success:
#     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
#     success,image = vidcap.read()
#     print('Read a new frame: ', success)
#     count += 1

# cv2.imshow("cam1", image1_raw)
# cv2.imshow("cam2", image2_raw)

im_out2 = cv2.flip(im_out2, flipCode=0)
stitched = np.hstack((im_out2[:, 0:450, :], im_out1[:, 450:, :]))

# cv2.imshow("cam1", im_out1)
# cv2.imshow("cam2", im_out2)
cv2.imshow('stitched', stitched)

cv2.waitKey(0)