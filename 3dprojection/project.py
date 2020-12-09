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
import importlib.util
import json

CALIB_DIR = "../data/salsa/calib"


def dynamic_load(source_path):
	spec = importlib.util.spec_from_file_location(
		f"dynamic_source_{source_path}", source_path
	)
	module_object = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module_object)
	return module_object


def get_oriented_boxes(depth_im, intrinsics, coords):
<<<<<<< HEAD
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
		pt_x1 = (
			(center_x - ints.intrinsic_matrix[0, 2]) * z1 / ints.intrinsic_matrix[0, 0]
		)
		pt_y1 = (
			(center_y - ints.intrinsic_matrix[1, 2]) * z1 / ints.intrinsic_matrix[1, 1]
		)
		mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
			size=0.05, origin=[-pt_x1, pt_y1, z1]
		)
		bounding_boxes.append(mesh_frame)

	return bounding_boxes


if __name__ == "__main__":
	DATASET = sys.argv[1]
	record = sys.argv[2]

	if DATASET == "lab":
		sources = [
			(
				os.path.abspath(os.path.join(CALIB_DIR, "cam1_lab.py")),
				[640, 480],
				os.path.abspath("../data/lab/test_images/lab.jpg"),
				os.path.abspath("../data/lab/test_depth/lab_depth.npy"),
			)
		]
	elif DATASET == "salsa":
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
		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
			color_im, depth_im, depth_scale=1, convert_rgb_to_intensity=False
		)

		geometries = []
		curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, ints)
		curr_pcd = curr_pcd.voxel_down_sample(voxel_size=0.001)
		curr_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
		geometries.append(curr_pcd)

		# os.chdir("/Users/Mokshith/Documents/launchpad/Watchman/3dprojection")

		frame_folder = "cam1" if DATASET == "salsa" else "lab"

		if DATASET == "salsa":
			frame_locations = sorted(
				os.listdir("./cam1_frames/"),
				key=lambda x: int(x.replace("frame", "").replace(".jpg", "")),
			)
		elif DATASET == "lab":
			frame_locations = sorted(
				os.listdir("./lab_frames/"),
				key = lambda x: int(x.replace(".jpg", ""))
			)
			frame_locations = frame_locations[0::10]

		coords = get_bounding_boxes(
			"cam1_frames/frame0.jpg" if DATASET == "salsa" else "lab_frames/003970.jpg"
		)
		bounding_boxes = (
			get_oriented_boxes(depth_im_temp, ints, coords)
			+ get_oriented_boxes(depth_im_temp, ints, coords)
			+ get_oriented_boxes(depth_im_temp, ints, coords)
		)
		for box in bounding_boxes:
			geometries.append(box)

		curr_loc = -1
		changer = 1

		camera_params = []

		for geom in geometries:
			vis.add_geometry(geom)
		cur_trajectory = o3d.io.read_pinhole_camera_trajectory(
			os.path.abspath("./{}_trajectory.json".format(DATASET))
		)

		print(
			"Trajectory:",
			cur_trajectory,
		)

		def move_forward(vis):
			global curr_loc
			global curr_pcd
			global bounding_boxes
			global changer
			
			curr_loc += 1

			if curr_loc % 5:
				return

			ctr = vis.get_view_control()

			if record == "play":
				ctr.convert_from_pinhole_camera_parameters(
					cur_trajectory.parameters[
						(curr_loc // 5) % (len(cur_trajectory.parameters))
					]
				)
			else:
				cur_params = ctr.convert_to_pinhole_camera_parameters()
				camera_params.append(cur_params)
				print(time.time())

				if curr_loc // 100 > 10:
					vis.register_animation_callback(None)
					print("Done sampling")
					traj = o3d.camera.PinholeCameraTrajectory()
					traj.parameters = camera_params
					o3d.io.write_pinhole_camera_trajectory(
						"./{}_trajectory.json".format(DATASET), traj
					)

			if curr_loc % (10 if DATASET == "salsa" else 20) == 0:
				coords = np.load(
					os.path.join(
						"{}_preds/".format(frame_folder),
						frame_locations[
							(curr_loc // (10 if DATASET == "salsa" else 20))
							% len(frame_locations)
						],
					).replace("jpg", "npy")
				)
				bounding_boxes_temp = get_oriented_boxes(depth_im_temp, ints, coords)

				for i in range(len(bounding_boxes_temp)):
					bounding_boxes[i].paint_uniform_color(np.array([0, 1, 0]))
					bounding_boxes[i].translate(
						bounding_boxes_temp[i].get_center(), relative=False
					)

				for i in range(len(bounding_boxes_temp), len(bounding_boxes)):
					bounding_boxes[i].paint_uniform_color(np.array([0, 0, 0]))
					bounding_boxes[i].translate(
						np.array([1e3, 1e3, 1e3]), relative=False
					)

				for geom in bounding_boxes:
					vis.update_geometry(geom)

	 

		vis.register_animation_callback(move_forward)
		vis.run()
		vis.destroy_window()
=======
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
        pt_x1 = (
            (center_x - ints.intrinsic_matrix[0, 2]) * z1 / ints.intrinsic_matrix[0, 0]
        )
        pt_y1 = (
            (center_y - ints.intrinsic_matrix[1, 2]) * z1 / ints.intrinsic_matrix[1, 1]
        )
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.025, origin=[pt_x1, -pt_y1, -z1]
        )
        bounding_boxes.append(mesh_frame)

    return bounding_boxes


if __name__ == "__main__":
    DATASET = sys.argv[1]
    record = sys.argv[2]

    if DATASET == "lab":
        sources = [
            (
                os.path.abspath(os.path.join(CALIB_DIR, "cam1_lab.py")),
                [1024, 768],
                os.path.abspath("../data/salsa/test_images/lab.jpg"),
                os.path.abspath("../data/salsa/test_depth/lab_depth.npy"),
            )
        ]
    elif DATASET == "salsa":
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
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_im, depth_im, depth_scale=1, convert_rgb_to_intensity=False
        )

        geometries = []
        curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, ints)
        curr_pcd = curr_pcd.voxel_down_sample(voxel_size=0.001)
        curr_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        geometries.append(curr_pcd)

        # os.chdir("/Users/Mokshith/Documents/launchpad/Watchman/3dprojection")

        frame_folder = "cam1" if DATASET == "salsa" else "lab"

        if DATASET == "salsa":
            frame_locations = sorted(
                os.listdir("./cam1_frames/"),
                key=lambda x: int(x.replace("frame", "").replace(".jpg", "")),
            )
        elif DATASET == "lab":
            frame_locations = sorted(os.listdir("./{}_frames/".format(frame_folder)))
            # frame_locations = frame_locations[0::10]

        coords = get_bounding_boxes(
            "cam1_frames/frame0.jpg" if DATASET == "salsa" else "lab_frames/003200.jpg"
        )
        bounding_boxes = (
            get_oriented_boxes(depth_im_temp, ints, coords)
            + get_oriented_boxes(depth_im_temp, ints, coords)
            + get_oriented_boxes(depth_im_temp, ints, coords)
        )
        for box in bounding_boxes:
            geometries.append(box)

        curr_loc = -1
        changer = 1

        camera_params = []

        for geom in geometries:
            vis.add_geometry(geom)
        cur_trajectory = o3d.io.read_pinhole_camera_trajectory(
            os.path.abspath("./{}_trajectory.json".format(DATASET))
        )

        print(
            "Trajectory:",
            cur_trajectory,
        )

        def ping_pong(idx):
            double_index = idx % (2 * len(cur_trajectory.parameters) - 1)
            midpoint = len(cur_trajectory.parameters) - 1
            return midpoint - abs(midpoint - double_index)

        def move_forward(vis):
            global curr_loc
            global curr_pcd
            global bounding_boxes
            global changer

            curr_loc += 1
            if curr_loc % 5:
                return

            ctr = vis.get_view_control()

            if record == "play":
                ctr.convert_from_pinhole_camera_parameters(
                    cur_trajectory.parameters[ping_pong(curr_loc // 5)]
                )
            else:
                cur_params = ctr.convert_to_pinhole_camera_parameters()
                camera_params.append(cur_params)
                print(time.time())

                if curr_loc // 100 > 10:
                    vis.register_animation_callback(None)
                    print("Done sampling")
                    traj = o3d.camera.PinholeCameraTrajectory()
                    traj.parameters = camera_params
                    o3d.io.write_pinhole_camera_trajectory(
                        "./{}_trajectory.json".format(DATASET), traj
                    )

            if curr_loc % (10 if DATASET == "salsa" else 20) == 0:
                coords = np.load(
                    os.path.join(
                        "{}_preds/".format(frame_folder),
                        frame_locations[
                            (curr_loc // (10 if DATASET == "salsa" else 20))
                            % len(frame_locations)
                        ],
                    ).replace("jpg", "npy")
                )
                bounding_boxes_temp = get_oriented_boxes(depth_im_temp, ints, coords)

                for i in range(len(bounding_boxes_temp)):
                    bounding_boxes[i].paint_uniform_color(np.array([0.15, 0.15, 1]))
                    bounding_boxes[i].translate(
                        bounding_boxes_temp[i].get_center(), relative=False
                    )

                for i in range(len(bounding_boxes_temp), len(bounding_boxes)):
                    bounding_boxes[i].paint_uniform_color(np.array([0, 0, 0]))
                    bounding_boxes[i].translate(
                        np.array([1e3, 1e3, 1e3]), relative=False
                    )

                for geom in bounding_boxes:
                    vis.update_geometry(geom)

            curr_loc += 1

        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()
>>>>>>> 1cc644bc6305a32ec61adc61b554ff222f0fb8d9
