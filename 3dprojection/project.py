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
import time

CALIB_DIR = "../data/salsa/calib"
DATASET = 'salsa'

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
        z1 = depth_im[int(center_y)][int(center_x)]
        pt_x1 = (center_x - ints.intrinsic_matrix[0, 2]) * z1 / ints.intrinsic_matrix[0, 0]
        pt_y1 = (center_y - ints.intrinsic_matrix[1, 2]) * z1 / ints.intrinsic_matrix[1, 1]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[-pt_x1, pt_y1, z1])
        bounding_boxes.append(mesh_frame)
        
    return bounding_boxes


if __name__ == "__main__":
    if DATASET == 'lab':
        sources = [
            (
                os.path.abspath(os.path.join(CALIB_DIR, "cam1_lab.py")),
                [1024, 768],
                os.path.abspath("../data/salsa/test_images/lab.jpg"),
                os.path.abspath("../data/salsa/test_depth/lab_depth.npy"),
            )
        ]
    elif DATASET == 'salsa':
        sources = [
            (
                os.path.abspath(os.path.join(CALIB_DIR, "cam1.py")),
                [1024, 768],
                os.path.abspath("../data/salsa/test_images/0_1.jpg"),
                os.path.abspath("../data/salsa/test_depth/0_1_disp.npy"),
            )
        ]


    vis = o3d.visualization.Visualizer()
    vis.create_window()

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

        print(eval('f"' + img + '"'))
        color_im = cv2.imread(eval('f"' + img + '"'))
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
        color_im = o3d.geometry.Image(color_im)

        depth_im = np.load(eval('f"' + depth_map + '"'))
        depth_im_temp = depth_im
        depth_im = o3d.geometry.Image(depth_im)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im, depth_im, depth_scale = 1, convert_rgb_to_intensity = False)

        curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, ints)
        curr_pcd = curr_pcd.voxel_down_sample(voxel_size=0.001)
        curr_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        vis.add_geometry(curr_pcd)

        os.chdir("/Users/Mokshith/Documents/launchpad/Watchman/3dprojection")
        
        frame_folder = 'cam1' if DATASET == 'salsa' else 'lab'

        if DATASET == 'salsa':
            frame_locations = sorted(os.listdir("./cam1_frames/"), key = lambda x: int(x.replace("frame", "").replace(".jpg", "")))
        elif DATASET == 'lab':
            frame_locations = sorted(os.listdir("./{}_frames/".format(frame_folder)))
            frame_locations = frame_locations[0::10]
        
        coords = get_bounding_boxes('cam1_frames/frame0.jpg' if DATASET == 'salsa' else 'lab_frames/003200.jpg')
        bounding_boxes = get_oriented_boxes(depth_im_temp, ints, coords) + get_oriented_boxes(depth_im_temp, ints, coords) + get_oriented_boxes(depth_im_temp, ints, coords)
        for box in bounding_boxes:
            vis.add_geometry(box)
        
        curr_loc = 0
        changer = 1

        def move_forward(vis):
            global curr_loc
            global curr_pcd
            global bounding_boxes
            global changer

            ctr = vis.get_view_control()

            if curr_loc % 300 == 0:
                changer *= -1
            
            if curr_loc == 0:
                ctr.rotate(100, 0)
            else:
                ctr.rotate(changer * 1.0, 0.0)

            if curr_loc % (10 if DATASET == 'salsa' else 10) == 0:
                coords = np.load(os.path.join("{}_preds/".format(frame_folder), frame_locations[curr_loc//(10 if DATASET == 'salsa' else 10)]).replace("jpg", "npy"))
                bounding_boxes_temp = get_oriented_boxes(depth_im_temp, ints, coords)

                for i in range(len(bounding_boxes_temp)):
                    bounding_boxes[i].paint_uniform_color(np.array([0, 1, 0]))
                    bounding_boxes[i].translate(bounding_boxes_temp[i].get_center(), relative = False)
                
                for i in range(len(bounding_boxes_temp), len(bounding_boxes)):
                    bounding_boxes[i].paint_uniform_color(np.array([0, 0, 0]))
                    bounding_boxes[i].translate(np.array([1, 1, 1]), relative = False)
                
                for geom in bounding_boxes:
                    vis.update_geometry(geom)
                
            
            curr_loc += 1
        
        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()