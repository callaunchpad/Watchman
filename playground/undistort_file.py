import cv2 as cv
import numpy as np
import argparse

#Getting the input file name, output file name, and calibration file name

parser = argparse.ArgumentParser()

parser.add_argument("input_image_name", type=str, help="name of input image (no need for .jpg)")
parser.add_argument("output_image_name", type=str, help="name of output image (no need for .jpg)")
parser.add_argument("calibration_file_name", type=str, help="name of the calibration file (NEEDS EXTENSION .calib)")

args = parser.parse_args()

#Parsing the calibration file

fin = open(args.calibration_file_name)

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

img = cv.imread(args.input_image_name + '.jpg')

#Camera matrix consisting of intrinsic camera parameters

camera_matrix = np.array([np.array([mv*f, 0, u0]),
                np.array([0, mu*f, v0]),
                np.array([0, 0, 1])])

#Array consisting of distortion parameters

dist_coeffs = np.array([k1, k2, p1, p2])

#Undistorting the image

dst = cv.undistort(img, camera_matrix, dist_coeffs)

#Writing the image

cv.imwrite(args.output_image_name + '.jpg', dst)

