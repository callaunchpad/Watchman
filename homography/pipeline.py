import cv2
import os
from glob import glob
import numpy as np
import bounding

pts_src1 = np.array([[614, 294], [581, 326],[503, 407], [313, 348], [478, 345]])
pts_src2 = np.array([[250, 279], [324, 281],[472, 286], [440, 437], [383, 325]])
pts_src3 = np.array([[127, 650], [186, 538], [265, 389],[542, 400], [333, 458]])
pts_src4 = np.array([[704, 314], [635, 292], [515, 257],[615, 192], [607, 248]])
source_points = [pts_src1, pts_src2, pts_src3, pts_src4]

pts_dst = np.array([[650, 550], [600, 550], [500, 550],[500, 400], [550, 500]]) #these are the zoomed out destination points

def main():
    caps = [cv2.VideoCapture(f"video/salsa_ps_cam{i}.avi") for i in range(1, 5)]
    homos = [cv2.findHomography(source_points[i], pts_dst)[0] for i in range(4)]
    total_frames = int(caps[0].get(7))
    interval = 30
    for i in range(total_frames):
        angles = [cap.read()[1] for cap in caps]
        if i % interval == 0:
            undis = []
            for j in range(4):
                with open(f"video/cam{j+1}.calib") as f:
                    undis.append(undistort(angles[j], f))
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

    result[:, 512:] = frames[0][:, 512:]
    #result[350:, 250:550] = frames[1][350:, 250:550]
    result[:, :512] = frames[3][:, :512]
    result[:, :200] = frames[2][:, :200]
    result[200:600, :512] = frames[2][200:600, :512]

    cv2.imwrite("out/stitched.png", result)

def get_angles(frame_num):
    frames = []
    for i in range(4):
        cap = cv2.VideoCapture(f"video/salsa_ps_cam{i+1}.avi")
        j = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if j == frame_num:

                with open(f"video/cam{i+1}.calib") as f:
                    undis = undistort(frame, f)
                    undis = cv2.polylines(undis, [source_points[i]], True, (0, 255, 0))
                    h, status = cv2.findHomography(source_points[i], pts_dst)
                    im_out = cv2.warpPerspective(undis, h, (undis.shape[1], undis.shape[0]))
                    frames.append(im_out)
                break
            j += 1

    return frames

def undistort(img, fin):
    current_line = ""

    while len(current_line) < 2 or current_line[0] != 'f':
        current_line = fin.readline()
    f = float(current_line[2:])

    while len(current_line) < 2 or current_line[:2] != 'mu':
        current_line = fin.readline()
    mu = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'mv':
        current_line = fin.readline()
    mv = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'u0':
        current_line = fin.readline()
    u0 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'v0':
        current_line = fin.readline()
    v0 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k1':
        current_line = fin.readline()
    k1 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k2':
        current_line = fin.readline()
    k2 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'k3':
        current_line = fin.readline()
    k3 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'p1':
        current_line = fin.readline()
    p1 = float(current_line[3:])

    while len(current_line) < 2 or current_line[:2] != 'p2':
        current_line = fin.readline()
    p2 = float(current_line[3:])

    #Camera matrix consisting of intrinsic camera parameters

    camera_matrix = np.array([np.array([mv*f, 0, u0]),
                    np.array([0, mu*f, v0]),
                    np.array([0, 0, 1])])

    #Array consisting of distortion parameters

    dist_coeffs = np.array([k1, k2, p1, p2])

    #Undistorting the image

    dst = cv2.undistort(img, camera_matrix, dist_coeffs)

    return dst

if __name__ == "__main__":
    main()