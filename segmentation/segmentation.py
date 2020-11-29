# Required Imports
import os
import numpy as np
from PIL import Image
from detect import detect
from depth import generate_depth_maps
import imageio
import argparse


def read_images(img_files):
    imgs = []
    for img_file in img_files:
        img_arr = imageio.imread(img_file)
        imgs.append(img_arr)
    return imgs


def form_room_naive(img_files, height, width):
    # Initialize blank image array and depth map
    img_arr = 255 * np.ones((height, width, 3), np.uint8)
    depth_map = np.zeros((height, width, 3), np.float32)

    # Create an unseen set for pixels we don't have room data about
    unseen = set()
    for h in range(height):
        for w in range(width):
            unseen.add((h, w))
    
    # Generate image arrays and run segmentation and depth models
    imgs = read_images(img_files)
    frames = detect(img_files)
    depths = generate_depth_maps(img_files)

    # Iterate through each frame and fill in gaps
    i = 0
    while unseen and i < len(img_files):
        print(img_files[i])
        img, frame, depth = imgs[i], frames[i], depths[i]
        for h, w in unseen.copy():
            color = frame[h][w]
            if color != [150, 5, 61]:
                img_arr[h][w] = img[h][w]
                depth_map[h][w] = depth[h][w]
                unseen.remove((h, w))
        i += 1 
    
    return img_arr, depth_map


def form_room_average(img_files, height, width):
    # Initialize blank image and depth map
    img_arr = np.zeros((height, width, 3), np.uint8)
    depth_map = np.zeros((height, width, 3), np.float32)

    # Generate image arrays and run segmentation and depth models
    imgs = read_images(img_files)
    frames = detect(img_files)
    depths = generate_depth_maps(img_files)

    # Keep track of counts for each pixel to average later
    counts = np.zeros((height, width), np.uint8)
    for i in range(len(img_files)):
        img, frame, depth = imgs[i], frames[i], depths[i]
        for h in range(height):
            for w in range(width):
                color = frame[h][w]
                if color != [150, 5, 61]:
                    img_arr[h][w] += img[h][w]
                    depth_map[h][w] += depth[h][w]
                    counts[h][w] += 1

    # Take the average by diving by counts
    for h in range(height):
        for w in range(width):
            if counts[h][w]:
                # print(img_arr[h][w])
                # print(counts[h][w])
                img_arr[h][w] = np.true_divide(img_arr[h][w], counts[h][w])
                depth_map[h][w] /= counts[h][w]

    return img_arr, depth_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create room.')
    parser.add_argument('images', type=str, help='Directory of images.')
    parser.add_argument('height', type=int, help='Height of image.')
    parser.add_argument('width', type=int, help='Width of image.')
    args = parser.parse_args()
    img_files = [args.images + x for x in os.listdir(args.images)]
    image_arr, depth_map = form_room_average(img_files, args.height, args.width)
    if not os.path.isdir('output'):
	    os.mkdir('output')
    imageio.imwrite('output/room.jpg', image_arr)
    np.save('output/room_depth.npy', depth_map)
