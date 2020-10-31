import cv2
import numpy as np
import detect

bounding_boxes = detect.get_bounding_boxes("frame3.jpg")

test = cv2.imread('frame3.jpg')

bounding_boxes = [np.array(x, dtype=int) for x in bounding_boxes]
print(bounding_boxes[0][0])
# print(np.array([540.0, 335.0]))

result = cv2.polylines(test, bounding_boxes, True, (0, 255, 0))
cv2.imwrite('result.png', result)