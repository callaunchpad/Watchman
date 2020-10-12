import cv2
import os
from glob import glob
import numpy as np

frames = [cv2.imread(f'frame{i}.jpg') for i in range(1, 5)]

pts_src1 = np.array([[433, 385], [501, 407], [453, 458],[382, 430]])
pts_src2 = np.array([[464, 326], [471, 286], [545, 288],[544, 331]])
pts_src3 = np.array([[359, 392], [274, 388], [298, 338],[376, 341]])
pts_src4 = np.array([[552, 236], [514, 258], [465, 243],[501, 223]])

source_points = [pts_src1, pts_src2, pts_src3, pts_src4]

pts_dst = np.array([[300, 500], [360, 500], [360, 560],[300, 560]])

for i in range(len(frames)):
    frames[i] = cv2.polylines(frames[i], [source_points[i]], True, (0, 255, 0))

    # find homography and warp perspective
    h, status = cv2.findHomography(source_points[i], pts_dst)
    im_out = cv2.warpPerspective(frames[i], h, (frames[i].shape[1], frames[i].shape[0]))

    cv2.imwrite(f"annotated{i}.png", frames[i])
    cv2.imwrite(f'result{i}.png', im_out)