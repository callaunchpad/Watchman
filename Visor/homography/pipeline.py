import cv2
import os
from glob import glob
import numpy as np
import bounding

pts_src1 = np.array([[345, 259],[371, 273], [347, 289], [319, 274]])
pts_src2 = np.array([[370, 297], [347, 320], [315, 301], [339, 281]])
pts_src3 = np.array([[209, 318], [191, 302], [219, 289],[238, 303]])
pts_src4 = np.array([[331, 241], [351, 230], [374, 241],[354, 253]])
source_points = [pts_src1, pts_src2, pts_src3, pts_src4]

pts_dst = np.array([[350, 350], [300, 350], [300, 300],[350, 300]]) #these are the zoomed out destination points

def main():
    # caps = [cv2.VideoCapture(f"video/cam{i}/%06d/") for i in range(1, 5)]
    # print(source_points[0])
    # print(pts_dst)
    # print(cv2.findHomography(source_points[0], pts_dst))
    homos = [cv2.findHomography(source_points[i], pts_dst)[0] for i in range(4)]
    # print(homos)
    paths = [f"video/cam{i}" for i in range(1, 5)]
    caps = []
    one_cam = []
    for i in paths:
        total_frames = 0
        cam_paths = [i + "/00000" + str(k) for k in range(3, 4)]
        for j in cam_paths:
            val = 0
            for im in sorted(os.listdir(j)):
                if val % 30 != 0:
                    val += 1
                    continue
                print(j + "/" + im)
                one_cam.append(cv2.imread(j + "/" + im))
                total_frames += 1
                val += 1
        caps.append(np.array(one_cam))
        one_cam = []
        print('done')

    # print(caps)
    
    interval = 30
    for i in range(total_frames):
        # print(i)
        # angles = [cap.read()[1] for cap in caps]
        # print(len(caps), caps[0].shape)
        angles = caps
        if i % interval == 0:
            undis = []
            for j in range(4):
                with open(f"video/cam{j+1}.pm") as f:
                    undis.append(undistort(angles[j][i], f))
            for j, undis_img in enumerate(undis):
                cv2.imwrite(f'out/undis{j}.png', undis_img)
            boxes = [get_boxes(f'out/undis{j}.png', homos[j]) for j in range(4)]
            outs = [cv2.warpPerspective(undis[j], homos[j], (undis[j].shape[1], undis[j].shape[0])) for j in range(4)]
            outs = [cv2.polylines(outs[j], boxes[j], True, (0, 255, 0)) for j in range(4)]
            cv2.imwrite(f"gif/stitched{i}.png", create_frame(outs))
            print(f'FINISHED FRAME {i}/{total_frames}')

def create_frame(frames):
    for i, frame in enumerate(frames):
        cv2.imwrite(f"out/frame{i}.png", frame)
    
    result = np.zeros(frames[0].shape)

    result[:, 350:] = frames[0][:, 350:]
    #result[350:, 250:550] = frames[1][350:, 250:550]
    result[:, :350] = frames[3][:, :350]
    #result[:, :200] = frames[2][:, :200]
    #result[200:600, :512] = frames[2][200:600, :512]

    return result

def get_boxes(name, homo):
    boxes = bounding.get_boxes(name)
    # print(boxes)
    # print(homo)
    transformed = np.array([cv2.perspectiveTransform(box.reshape(-1, 1, 2).astype(np.float32), homo) for box in boxes], dtype=int)
    return transformed

def undistort(img, fin):
    # print('test1')
    current_line = ""

    while len(current_line) < 2 or current_line[:2] != 'fx':
        # print(current_line)
        # print('break')
        current_line = fin.readline()
    fx = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'fy':
        current_line = fin.readline()
    fy = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'cx':
        current_line = fin.readline()
    cx = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'cy':
        current_line = fin.readline()
    cy = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k1':
        current_line = fin.readline()
    k1 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k2':
        current_line = fin.readline()
    k2 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k3':
        current_line = fin.readline()
    k3 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k4':
        current_line = fin.readline()
    k4 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k5':
        current_line = fin.readline()
    k5 = float(current_line[3:])

    #Camera matrix consisting of intrinsic camera parameters

    camera_matrix = np.array([
                        np.array([fx, 0, cx]),
                        np.array([0, fy, cy]),
                        np.array([0, 0, 1])
                        ])

    #Array consisting of distortion parameters

    dist_coeffs = np.array([k1, k2, 0, 0])

    #Undistorting the image

    img = np.array([np.array(i) for i in img])

    dst = cv2.undistort(img, camera_matrix, dist_coeffs)

    return dst

if __name__ == "__main__":
    main()