from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
from XMem.inference.interact.interactive_utils import overlay_davis
from config import DEVICE
from tools.mask_display import visualize_unique_mask, visualize_wb_mask
import torch
from tools.mask_merge import create_mask, merge_masks
from tools.overlay_image import show_mask


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

    def predict(self, prompt, mode='point', multimask=True):

        assert self.embedded, 'dont set image'
        assert mode in ['point', 'box'], 'mode can be point, box'

        if mode == 'point':
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt['point_coords'],
                point_labels=prompt['point_labels'],
                box=prompt['boxes'],
                multimask_output=multimask,
            )
        elif mode == 'box':
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt['point_coords'],
                point_labels=prompt['point_labels'],
                box=prompt['boxes'],
                multimask_output=multimask,
            )
        else:
            raise ('Error')

        return masks, scores, logits


if __name__ == '__main__':
    path = 'video-test/truck.jpg'
    # path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()

    bboxes = [[476, 166, 102, 154], [8, 252, 91, 149], [106, 335, 211, 90]]
    points = [[531, 230], [45, 321], [226, 360], [194, 313]]

    prompts = {
        'mode': 'point',
        'point_coords': [[531, 230], [45, 321], [226, 360], [194, 313]],
        'point_labels': [1] * len(points),
        'boxes': None,
    }

    prompts = {
        'mode': 'box',
        'point_coords': [None, None, None, None],
        'point_labels': [None, None, None, None],
        'boxes': [
            [476, 166, 578, 320],
            [8, 252, 99, 401],
            [106, 335, 317, 425],
            [155, 283, 225, 339],
        ],
    }

    prompts = {
        'mode': 'box',
        'point_coords': [[575, 750]],
        'point_labels': [0],
        'boxes': [[425, 600, 700, 875]],
    }

    # prompts = {
    #     'mode': 'sig',
    #     'point_coords': None,
    #     'point_labels': None,
    #     'boxes': [
    #         [75, 275, 1725, 850],
    #         [425, 600, 700, 875],
    #         [1375, 550, 1650, 800],
    #         [1240, 675, 1400, 750],
    #     ],
    # }

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg = Segmenter2()
    seg.set_image(frame)

    maskss = []
    if prompts['mode'] == 'point':
        for point_c, point_l in zip(prompts['point_coords'], prompts['point_labels']):
            prompt = {
                'point_coords': np.array([point_c]),
                'point_labels': np.array([point_l]),
                'boxes': None,
            }
            masks, scores, logits = seg.predict(prompt, prompts['mode'])
            maskss.append(masks[np.argmax(scores)])
    elif prompts['mode'] == 'box':
        for point_c, point_l, box in zip(
            prompts['point_coords'], prompts['point_labels'], prompts['boxes']
        ):
            prompt = {
                'point_coords': None if point_c is None else np.array([point_c]),
                'point_labels': None if point_l is None else np.array([point_l]),
                'boxes': np.array([box]),
            }
            masks, scores, logits = seg.predict(prompt, prompts['mode'], multimask=False)
            maskss.append(masks[np.argmax(scores)])
    # else:
    #     prompts = {
    #         'mode': 'box',
    #         'point_coords': None,
    #         'point_labels': None,
    #         'boxes': [
    #             [476, 166, 578, 320],
    #             [8, 252, 99, 401],
    #             [106, 335, 317, 425],
    #             [155, 283, 225, 339],
    #         ],
    #     }
    #     masks, scores, logits = seg.predict(prompts, mode='box', multimask=False)

    print(len(maskss))
    # plt.imshow(frame)
    ma = []
    for mask in maskss:
        # mask = show_mask(mask.squeeze(0), plt.gca(), random_color=True)
        mask = create_mask(mask, random_color=True)
        ma.append(mask)
    # plt.axis('off')
    # plt.show()
    # input_box = np.array([425, 600, 700, 875])
    # input_point = np.array([[575, 750]])
    # input_label = np.array([0])
    # show_masks(
    #     frame,
    #     masks,
    #     scores,
    #     box_coords=input_box,
    #     point_coords=input_point,
    #     input_labels=input_label,
    # )
    mask, unique_mask = merge_masks(maskss)
    f = overlay_davis(frame, unique_mask)
    mask = visualize_unique_mask(unique_mask)
    cv2.imshow('asd', mask)
    cv2.imshow('asd', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
