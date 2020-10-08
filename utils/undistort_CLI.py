import cv2 as cv
import numpy as np
import argparse

#Creating the command line parser

parser = argparse.ArgumentParser()

parser.add_argument("input_image_name", type=str, help="name of input image (no need for .jpg)")
parser.add_argument("output_image_name", type=str, help="name of output image (no need for .jpg)")
parser.add_argument("f", type=float, help="focal length of camera, f")
parser.add_argument("mu", type=float, help="camera instrinic, mu")
parser.add_argument("mv", type=float, help="camera instrinic, mv")
parser.add_argument("u0", type=float, help="camera instrinic, u0")
parser.add_argument("v0", type=float, help="camera instrinic, v0")
parser.add_argument("k1", type=float, help="distortion parameter, k1")
parser.add_argument("k2", type=float, help="distortion parameter, k2")
parser.add_argument("k3", type=float, help="distortion parameter, k3")
parser.add_argument("p1", type=float, help="distortion parameter, p1")
parser.add_argument("p2", type=float, help="distortion parameter, p2")

args = parser.parse_args()

#Intrinsic camera parameters

f = args.f
mu = args.mu
mv = args.mv

u0 = args.u0
v0 = args.v0

#Distortion Parameters

k1 = args.k1
k2 = args.k2
k3 = args.k3
p1 = args.p1
p2 = args.p2

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

