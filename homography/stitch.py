import cv2
import os
from glob import glob
import numpy as np



frames = [cv2.imread(f'result{i}.png') for i in range(4)]

result = np.zeros(frames[0].shape)

result[:, 512:] = frames[0][:, 512:]
#result[350:, 250:550] = frames[1][350:, 250:550]
result[:, :512] = frames[3][:, :512]
#result[200:600, :512] = frames[2][200:600, :512]

# frames[1][:580, :630, :] = frames[0][:580, :630, :]
# frames[1][450:, :611, :] = frames[2][450:, :611, :]
# frames[1][260:, :788, :] = frames[3][260:, :788, :]

cv2.imwrite("stitched.png", result)