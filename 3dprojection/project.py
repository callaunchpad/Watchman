# Required Imports
import sys
sys.path.append("../")
import os
import numpy as np
import open3d as o3d
import cv2
import argparse
import importlib.util
from yolo.detect import get_bounding_boxes


CALIB_DIR = "../data/salsa/calib"


def dynamic_load(source_path):
    spec = importlib.util.spec_from_file_location(
        f"dynamic_source_{source_path}", source_path
    )
    module_object = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_object)
    return module_object

def get_oriented_boxes(depth_im, intrinsics, coords):
    bounding_boxes = []
    for i in range(len(coords)):
        bounding_box_coords = o3d.utility.Vector3dVector([])
        points = []
        top_left = coords[i][0]
        bottom_right = coords[i][3]
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        height = top_left[1] - bottom_right[1]
        width = top_left[0] - bottom_right[0]
        # coords[i] = [(center_x - width//8, center_y - height//8), (center_x + width//8, center_y - height//8), (center_x - width//8, center_y + height//8), (center_x + width//8, center_y + height//8)]
        # new_top_left = (top_left[0] + ((top_left[0] - bottom_right[0]) // 4), top_left[1] + ((top_left[1] - bottom_right[1]) // 4))
        # new_bottom_right = (bottom_right[0] - ((top_left[0] - bottom_right[0]) // 4), bottom_right[1] - ((top_left[1] - bottom_right[1]) // 4))
        # coords[i] = []
        for j in range(len(coords[i])):
            x1, y1 = coords[i][j]
            z1 = depth_im[int(center_y)][int(center_x)]
            pt_x1 = (x1 - ints.intrinsic_matrix[0, 2]) * z1 / ints.intrinsic_matrix[0, 0]
            pt_y1 = (y1 - ints.intrinsic_matrix[1, 2]) * z1 / ints.intrinsic_matrix[1, 1]
            pt_z1 = z1
            bounding_box_coords.extend([[pt_x1, pt_y1, pt_z1], [pt_x1, pt_y1, pt_z1 + 0.00005]])
        bounding_boxes.append(o3d.geometry.OrientedBoundingBox.create_from_points(bounding_box_coords))

    


    return bounding_boxes


if __name__ == "__main__":
    sources = [
        (
            os.path.join(CALIB_DIR, "cam1.py"),
            [1024, 768],
            "../data/salsa/test_images/0_1.jpg",
            "../data/salsa/test_depth/0_1_disp.npy",
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

    pcds = []
    bboxes = []
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
        ints = o3d.camera.PinholeCameraIntrinsic()
        ints.set_intrinsics(*intrinsics)
        extrinsics = module_object.extrinsics

        color_im = cv2.imread(eval('f"' + img + '"'))
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
        color_im = o3d.geometry.Image(color_im)

        depth_im = np.load(eval('f"' + depth_map + '"'))

        coords = get_bounding_boxes(img)
        bounding_boxes = get_oriented_boxes(depth_im, ints, coords)

        depth_im = o3d.geometry.Image(depth_im)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im, depth_im, depth_scale = 1, convert_rgb_to_intensity = False)

        curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, ints)
        curr_pcd = curr_pcd.voxel_down_sample(voxel_size=0.001)
        pcds.append(curr_pcd)
        
        bboxes.extend(bounding_boxes)
    
    pcds.extend(bboxes)
    print(pcds)
    o3d.visualization.draw_geometries(pcds)
