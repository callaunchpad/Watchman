import cv2
import os
import argparse

def capture_frames(video_path, output_path, frames=float('inf')):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    while count < frames and vidcap.isOpened():
        success, frame = vidcap.read()
        if success:
            print("hello")
            cv2.imwrite(os.path.join(output_path, '%d.jpg') % count, frame)
            count += 1
        else:
            print("bye")
            break
    cv2.destroyAllWindows()
    vidcap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture video frames.')
    parser.add_argument('--video')
    parser.add_argument('--out')
    parser.add_argument('--frames', type=int)
    
    args = parser.parse_args()
    video_path = args.video
    output_path = args.out
    frames = args.frames

    capture_frames(video_path, output_path, frames)