import utils
import cv2
import argparse
import os

def get_frames(video_file, output_path, camera_number):				#Input the path to the video (INCLUDING THE FILE NAME) and the directory to output the frames to.
	vidcap = cv2.VideoCapture(video_file)
	success,image = vidcap.read()
	count = 0
	success = True
	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	while success and count <= 6000:
		success,image = vidcap.read()
		if count % 480 == 0:
			image = utils.undistort_image(image, camera_number)
			cv2.imwrite(output_path + "/frame%d.jpg" % count, image)		# save frame as JPEG file
			if cv2.waitKey(10) == 27:						# exit if Escape is hit
				break
		print(count)
		count += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extract video frames.')
	parser.add_argument('video', type=str, help='Path to video.')
	parser.add_argument('camera', type=str, help='Camera number.')
	parser.add_argument('--output', default='./', type=str, help='Output path.')
	args = parser.parse_args()
	get_frames(args.video, args.output, args.camera)
