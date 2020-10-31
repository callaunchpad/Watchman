# Required Imports
import os
import numpy as np
import open3d as o3d
import cv2
import argparse
import importlib.util
from detect import get_bounding_boxes

CALIB_DIR = "../data/salsa/calib"


def dynamic_load(source_path):
    spec = importlib.util.spec_from_file_location(
        f"dynamic_source_{source_path}", source_path
    )
    module_object = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_object)
    return module_object


def project(img, disp_map, intrinsics, coords):
    # Set camera parameters for reference in later calculations

    ints = o3d.camera.PinholeCameraIntrinsic()
    ints.set_intrinsics(*intrinsics)

    # # Read color image
    # color_im = cv2.imread(img)
    # color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

    # Read disparity map
    disparity = np.load(disp_map)[0]
    depth_im = 1 / disparity  # unscaled disparity to depth conversion
    # print(depth_im.max(), depth_im.min(), depth_im.shape, color_im.shape)

    # pts = []
    # colors = []
    # for y in range(color_im.shape[0]):
    #     for x in range(color_im.shape[1]):
    #         z = depth_im[0][y][x]

    #         # x, y, z calculation in 3D space based on intrinsics and depth
    #         pt_x = (x - ints.intrinsic_matrix[0, 2]) * z / ints.intrinsic_matrix[0, 0]
    #         pt_y = (y - ints.intrinsic_matrix[1, 2]) * z / ints.intrinsic_matrix[1, 1]
    #         pt_z = z

    #         pts.append(np.array([pt_x, pt_y, pt_z]))
    #         colors.append(color_im[y][x] / 255)

    bounding_box_coords = o3d.utility.Vector3dVector([])
    for x1, y1, x2, y2 in coords:
        z1 = depth_im[0][int(y1.item()) - 1][int(x1.item()) - 1]
        # x, y, z calculation in 3D space based on intrinsics and depth
        pt_x1 = (x1 - ints.intrinsic_matrix[0, 2]) * z1 / ints.intrinsic_matrix[0, 0]
        pt_y1 = (y1 - ints.intrinsic_matrix[1, 2]) * z1 / ints.intrinsic_matrix[1, 1]
        pt_z1 = z1

        z2 = depth_im[0][int(y2.item()) - 1][int(x2.item()) - 1]
        # x, y, z calculation in 3D space based on intrinsics and depth
        pt_x2 = (x2 - ints.intrinsic_matrix[0, 2]) * z2 / ints.intrinsic_matrix[0, 0]
        pt_y2 = (y2 - ints.intrinsic_matrix[1, 2]) * z2 / ints.intrinsic_matrix[1, 1]
        pt_z2 = z2

        bounding_box_coords.extend([[pt_x1, pt_y1, pt_z1], [pt_x1, pt_y1, pt_z1 + 50], [pt_x2, pt_y2, pt_z2], [pt_x2, pt_y2, pt_z2 + 50]])
    bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bounding_box_coords)

        

    


    return bounding_box


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
            "0_1.jpg",
            "0_1_disp.npy",
        ),
        # (
        #     os.path.join(CALIB_DIR, "cam2.py"),
        #     [1024, 768],
        #     "0_{i+1}.jpg",
        #     "0_{i+1}_disp.npy",
        # ),
        # (
        #     os.path.join(CALIB_DIR, "cam3.py"),
        #     [1024, 768],
        #     "0_{i+1}.jpg",
        #     "0_{i+1}_disp.npy",
        # ),
        # (
        #     os.path.join(CALIB_DIR, "cam4.py"),
        #     [1024, 768],
        #     "0_{i+1}.jpg",
        #     "0_{i+1}_disp.npy",
        # ),
    ]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector([])
    pcd.colors = o3d.utility.Vector3dVector([])
    pairs = []
    bbox = None
    pcds = []
    for i, intr_tuple in enumerate(sources):
        print(i)
        conf_fname, resolution, img, depth_map = intr_tuple
        module_object = dynamic_load(conf_fname)
        intrinsics = module_object.intrinsics
        intrinsics = resolution + [
            intrinsics[1][1],
            intrinsics[0][0],
            intrinsics[0][-1],
            intrinsics[1][-1],
        ]
        color_im = cv2.imread(eval('f"' + img + '"'))
        print(os.listdir("."))
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
        color_im = o3d.geometry.Image(color_im)
        print(color_im)
        # Read disparity map
        disparity = np.load(eval('f"' + depth_map + '"'))[0][0]
        depth_im = 1 / disparity  # unscaled disparity to depth conversion
        depth_im = o3d.geometry.Image(depth_im)
        print(depth_im)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im, depth_im, depth_scale = 1, convert_rgb_to_intensity = False)
        extrinsics = module_object.extrinsics

        ints = o3d.camera.PinholeCameraIntrinsic()
        ints.set_intrinsics(*intrinsics)
        print("hello")
        curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, ints, extrinsic = np.linalg.inv(extrinsics))
        print(curr_pcd)
        pcds.append(curr_pcd)
        print("hehe")
        # print(img)
        coords = get_bounding_boxes(img)
        bounding_box = project(
            eval('f"' + img + '"'), eval('f"' + depth_map + '"'), intrinsics, coords
        )
        # extrinsics = module_object.extrinsics
        # cur_pcd = o3d.geometry.PointCloud()
        # cur_pcd.points = o3d.utility.Vector3dVector(pts)
        # cur_pcd.colors = o3d.utility.Vector3dVector(colors)
        # cur_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # cur_pcd.transform(extrinsics)
        # pcd.points.extend(cur_pcd.points)
        # pcd.colors.extend(cur_pcd.colors)
        bbox = bounding_box
    


    # print(np.concatenate((pts_1, pts_2)).shape)
    # pcd.points = o3d.utility.Vector3dVector(
    #     np.concatenate(tuple(pair[0] for pair in pairs))
    # )
    # pcd.colors = o3d.utility.Vector3dVector(
    #     np.concatenate(tuple(pair[1] for pair in pairs))
    # )

    # transform to view properly
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #bbox.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd.transform([])

    # Visualize in viewer
    pcds.append(bbox)
    o3d.visualization.draw_geometries(pcds)
