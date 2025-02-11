from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from config import DEVICE

sam2_checkpoint = 'sam2.1_hiera_large.pt'
model_cfg = 'sam2.1_hiera_l.yaml'

build = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)

sam = SAM2ImagePredictor(build)


path = 'video.mp4'
video = cv2.VideoCapture(path)
ret, frame = video.read()
frame_cop = frame.copy()
video.release()
bboxes = [(476, 166, 102, 154), (8, 252, 91, 149), (106, 335, 211, 90)]
points = [[531, 230], [45, 321], [226, 360], [194, 313]]
input_label = np.array([1])
sam.set_image(frame_cop)
maskss = []
for point in points:
    masks, scores, logits = sam.predict(
        point_coords=np.array([point]), point_labels=input_label, multimask_output=False
    )
    maskss.append(masks[0])
print(len(maskss[0]))
cv2.imshow('asd', maskss[3])
cv2.waitKey(0)
cv2.destroyAllWindows()
