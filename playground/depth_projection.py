# Required Imports
import numpy as np
import open3d as o3d
import cv2

# Set camera parameters for reference in later calculations
ints = o3d.camera.PinholeCameraIntrinsic()
ints.set_intrinsics(1024, 768, 701.195, 699.50371766, 485.301, 432.17)

# Read color image
color_im = cv2.imread('0.png')
color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

# Read disparity map
disparity = np.load('0_depth_scaled.npy')[0]
depth_im = 1 / disparity # unscaled disparity to depth conversion
print(depth_im.max(), depth_im.min(), depth_im.shape, color_im.shape)

pts = []
colors = []
for y in range(color_im.shape[0]):
    for x in range(color_im.shape[1]):
        z = depth_im[0][y][x]

        # x, y, z calculation in 3D space based on intrinsics and depth
        pt_x = (x - ints.intrinsic_matrix[0, 2]) * z / ints.intrinsic_matrix[0, 0]
        pt_y = (y - ints.intrinsic_matrix[1, 2]) * z / ints.intrinsic_matrix[1, 1]
        pt_z = z

        pts.append(np.array([pt_x, pt_y, pt_z]))
        colors.append(color_im[y][x] / 255)

pts = np.array(pts)
colors = np.array(colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

# transform to view properly
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Visualize in viewer
o3d.visualization.draw_geometries([pcd])
