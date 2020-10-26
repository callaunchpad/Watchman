# Required Imports
import os
import numpy as np
import open3d as o3d
import cv2
import argparse
import importlib.util

CALIB_DIR = "../data/salsa/calib"


def dynamic_load(source_path):
    spec = importlib.util.spec_from_file_location(
        f"dynamic_source_{source_path}", source_path
    )
    module_object = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_object)
    return module_object


def project(img, disp_map, intrinsics):
    # Set camera parameters for reference in later calculations

    ints = o3d.camera.PinholeCameraIntrinsic()
    ints.set_intrinsics(*intrinsics)

    # Read color image
    color_im = cv2.imread(img)
    color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

    # Read disparity map
    disparity = np.load(disp_map)[0]
    depth_im = 1 / disparity  # unscaled disparity to depth conversion
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str)
    # parser.add_argument("--disp", type=str)

    # args = parser.parse_args()
    # project(args.image, args.disp)
    sources = [
        (
            os.path.join(CALIB_DIR, "cam1.py"),
            [1024, 768],
            "0_{i+1}.jpg",
            "0_{i+1}_disp.npy",
        ),
        (
            os.path.join(CALIB_DIR, "cam2.py"),
            [1024, 768],
            "0_{i+1}.jpg",
            "0_{i+1}_disp.npy",
        ),
        (
            os.path.join(CALIB_DIR, "cam3.py"),
            [1024, 768],
            "0_{i+1}.jpg",
            "0_{i+1}_disp.npy",
        ),
        (
            os.path.join(CALIB_DIR, "cam4.py"),
            [1024, 768],
            "0_{i+1}.jpg",
            "0_{i+1}_disp.npy",
        ),
    ]

    pcd = o3d.geometry.PointCloud()
    pairs = []
    for i, intr_tuple in enumerate(sources):
        conf_fname, resolution, img, depth_map = intr_tuple
        module_object = dynamic_load(conf_fname)
        intrinsics = module_object.intrinsics
        intrinsics = resolution + [
            intrinsics[1][1],
            intrinsics[0][0],
            intrinsics[0][-1],
            intrinsics[1][-1],
        ]
        pairs.append(
            project(eval('f"' + img + '"'), eval('f"' + depth_map + '"'), intrinsics)
        )

    # print(np.concatenate((pts_1, pts_2)).shape)
    pcd.points = o3d.utility.Vector3dVector(
        np.concatenate(tuple(pair[0] for pair in pairs))
    )
    pcd.colors = o3d.utility.Vector3dVector(
        np.concatenate(tuple(pair[1] for pair in pairs))
    )

    # transform to view properly
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd.transform([])

    # Visualize in viewer
    o3d.visualization.draw_geometries([pcd])
