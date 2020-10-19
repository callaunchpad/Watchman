# Required Imports
import numpy as np
import open3d as o3d
import cv2
import argparse

def project(img, disp_map, intrinsics):
    # Set camera parameters for reference in later calculations

    ints = o3d.camera.PinholeCameraIntrinsic()
    ints.set_intrinsics(*intrinsics)

    # Read color image
    color_im = cv2.imread(img)
    color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

    # Read disparity map
    disparity = np.load(disp_map)[0]
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

    return pts, colors
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str)
    # parser.add_argument("--disp", type=str)

    # args = parser.parse_args()
    # project(args.image, args.disp)
    int_1 = [1024, 768, 701.195, 699.50371766, 485.301, 432.17]
    int_2 = [1024, 768, 695.313, 695.313 * 0.997147, 543.21, 375.81]
    int_3 = [1024, 768, 669.93, 669.93 * 0.997553, 509.605, 358.812]
    int_4 = [1024, 768, 674.319, 674.319 * 0.995272, 461.743, 398.09]

    pts_1, colors_1 = project('0_1.jpg', '0_1_disp.npy', int_1)
    pts_2, colors_2 = project('0_2.jpg', '0_2_disp.npy', int_2)
    pts_3, colors_3 = project('0_3.jpg', '0_3_disp.npy', int_3)
    pts_4, colors_4 = project('0_4.jpg', '0_4_disp.npy', int_4)
    
    pcd = o3d.geometry.PointCloud()
    # print(np.concatenate((pts_1, pts_2)).shape)
    pcd.points = o3d.utility.Vector3dVector(np.concatenate((pts_1, pts_2, pts_3, pts_4)))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate((colors_1, colors_2, colors_3, colors_4)))

    # transform to view properly
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Visualize in viewer
    o3d.visualization.draw_geometries([pcd])
