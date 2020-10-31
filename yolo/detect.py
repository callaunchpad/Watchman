import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect(given_source):
	bounding_boxes = []
	imgsz = 512  # (320, 192) or (416, 256) or (608, 352) for (height, width)
	out, source, weights, half, view_img, save_txt = "output", given_source, "weights/yolov3-spp-ultralytics.pt", False, False, False
	webcam = False

	# Initialize
	device = torch_utils.select_device("")

	# Initialize model
	model = Darknet(check_file("cfg/yolov3-spp.cfg"), imgsz)

	attempt_download(weights)
	model.load_state_dict(torch.load(weights, map_location=device)['model'])

	classify = False

	# Eval mode
	model.to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None
	if webcam:
		view_img = True
		torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreams(source, img_size=imgsz)
	else:
		save_img = True
		dataset = LoadImages(source, img_size=imgsz)

	# Get names and colors
	names = load_classes(check_file("data/coco.names"))
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

	# Run inference
	t0 = time.time()
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)	# init img
	_ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
	for path, img, im0s, vid_cap in dataset:
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		t1 = torch_utils.time_synchronized()
		pred = model(img, augment=False)[0]
		t2 = torch_utils.time_synchronized()

		# Apply NMS
		pred = non_max_suppression(pred, 0.3, 0.6,
								   multi_label=False, classes=None, agnostic=False)

		# Process detections
		for i, det in enumerate(pred):	# detections for image i
			if webcam:	# batch_size >= 1
				p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
			else:
				p, s, im0 = path, '', im0s

			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]	#  normalization gain whwh
			if det is not None and len(det):
				# Rescale boxes from imgsz to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %ss, ' % (n, names[int(c)])  # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):
					label = '%s %.2f' % (names[int(cls)], conf)
					xyxy = np.array(xyxy)
					if label.startswith("person"):
						bounding_boxes.append(list(xyxy))
 
	return bounding_boxes
	print('Done. (%.3fs)' % (time.time() - t0))

def get_bounding_boxes(given_source):
	with torch.no_grad():
		return detect(given_source)
