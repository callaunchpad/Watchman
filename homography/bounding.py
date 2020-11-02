import cv2
import numpy as np
import detect

def foot_only(box):
    # print(box)
    dist = (box[2][0] - box[1][0])/3
    tl = list(map(int, [box[1][0], box[1][1] - dist]))
    tr = list(map(int, [box[2][0], box[2][1] - dist]))
    br = list(map(int, box[2]))
    bl = list(map(int, box[1]))
    return np.array([tl, bl, br, tr])
    
    

bounding_boxes = [detect.get_bounding_boxes(f"undis{i}.jpg") for i in range(1, 5)]

ims = [cv2.imread(f'undis{i}.jpg') for i in range(1, 5)]

# print("TESTING", len(bounding_boxes), len(bounding_boxes[0]))

bounding_boxes = [[foot_only(x) for x in bounding_boxes[i]] for i in range(4)]
# print(bounding_boxes[0][0])

for i in range(4):
    result = cv2.polylines(ims[i], bounding_boxes[i], True, (0, 255, 0), 2)
    cv2.imwrite(f'bounded{i+1}.png', result)