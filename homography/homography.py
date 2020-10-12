import cv2
import os
from glob import glob
import numpy as np

frames = [cv2.imread(f'undis{i}.jpg') for i in range(1, 5)]

pts_src1 = np.array([[614, 294], [581, 326],[503, 407], [313, 348], [478, 345]])
pts_src2 = np.array([[250, 279], [324, 281],[472, 286], [440, 437], [383, 325]])
pts_src3 = np.array([[127, 650], [186, 538], [265, 389],[542, 400], [333, 458]])
pts_src4 = np.array([[704, 314], [635, 292], [515, 257],[615, 192], [607, 248]])

source_points = [pts_src1, pts_src2, pts_src3, pts_src4]

#pts_dst = np.array([[750, 550], [650, 550], [450, 550],[450, 250], [550, 450]])
pts_dst = np.array([[650, 550], [600, 550], [500, 550],[500, 400], [550, 500]])

for i in range(len(frames)):
    frames[i] = cv2.polylines(frames[i], [source_points[i]], True, (0, 255, 0))

    # find homography and warp perspective
    h, status = cv2.findHomography(source_points[i], pts_dst)
    im_out = cv2.warpPerspective(frames[i], h, (frames[i].shape[1], frames[i].shape[0]))

    cv2.imwrite(f"annotated{i}.png", frames[i])
    cv2.imwrite(f'result{i}.png', im_out)