from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from config import DEVICE
from tools.mask_display import visualize_unique_mask, visualize_wb_mask
import torch
from tools.mask_merge import merge_masks


class Segmenter2:
    def __init__(self, device: str = DEVICE):
        self.device = device
        sam2_checkpoint = 'models/sam2_hiera_large.pt'
        model_cfg = 'sam2_hiera_l.yaml'
        build = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(build)
        self.embedded = False

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        if self.embedded:
            print('please reset_image')
            return
        self.predictor.set_image(image)
        self.embedded = True

    @torch.no_grad()
    def reset_image(self):
        self.predictor.reset_predictor()
        self.embedded = False

    def predict(self, prompts, mode='point', multimask=True):
        if mode == 'point':
            masks, scores, logits = self.predictor.predict(
                point_coords=prompts['point_coords'],
                point_labels=prompts['point_labels'],
                multimask_output=multimask,
            )
        return masks, scores, logits


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()

    bboxes = [(476, 166, 102, 154), (8, 252, 91, 149), (106, 335, 211, 90)]
    points = [[531, 230], [45, 321], [226, 360], [194, 313]]

    mode = 'point'
    prompts = {
        'point_coords': np.array([[531, 230], [45, 321], [226, 360], [194, 313]]),
        'point_labels': np.array([1] * len(points)),
    }
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg = Segmenter2()
    seg.set_image(frame)

    # masks, scores, logits = seg.predict(prompts, mode)
    # print(masks[np.argmax(scores)])
    # print(scores)
    # paint = visualize_wb_mask(masks[2])
    # cv2.imshow('paint', paint)
    maskss = []
    for point in points:
        prompts = {
            'point_coords': np.array([point]),
            'point_labels': np.array([1]),
        }
        masks, scores, logits = seg.predict(prompts, mode)
        maskss.append(masks[np.argmax(scores)])
    mask, unique_mask = merge_masks(maskss)
    mask = visualize_unique_mask(unique_mask)
    cv2.imshow('asd', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
