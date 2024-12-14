import cv2
import numpy as np
import psutil
import torch
from XMem.inference.inference_core import InferenceCore
from XMem.model.network import XMem
from config import XMEM_CONFIG, DEVICE
from torchvision import transforms
from XMem.util.range_transform import im_normalization
from XMem.inference.interact.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
    torch_prob_to_numpy_mask,
    overlay_davis,
)
from segmenter import Segmenter
from tools.mask_display import visualize_unique_mask, visualize_wb_mask, mask_map
from tools.contour_detector import getting_coordinates


class TrackerCore:
    def __init__(self, device: str = DEVICE):
        self.device = device
        if self.device.lower() != 'cpu':
            self.network = XMem(XMEM_CONFIG, 'models/XMem.pth').eval().to('cuda')
        else:
            self.network = XMem(
                XMEM_CONFIG, 'models/XMem.pth', map_location='cpu'
            ).eval()
        self.processor = InferenceCore(self.network, XMEM_CONFIG)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization
        ])

    @torch.no_grad()
    def track(self, frame: np.ndarray, mask: np.ndarray = None):
        frame_torch, _ = image_to_torch(frame, device=DEVICE)
        # frame_tensor = self.im_transform(frame).to(self.device)
        # mask = torch.Tensor(mask).to(self.device)

        # probs = self.processor.step(frame_torch, mask_torch, num_objects + 1)
        if mask is not None:
            num_objects = len(np.unique(mask)) - 1
            self.processor.set_all_labels(range(1, num_objects + 1))
            mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(DEVICE)
            probs = self.processor.step(frame_torch, mask_torch[1:])
        else:
            probs = self.processor.step(frame_torch)

        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
        return out_mask

    @torch.no_grad()
    def clear_memory(self):
        self.processor.clear_memory()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()
    bboxes = [(476, 166, 102, 154), (8, 252, 91, 149), (106, 335, 211, 90)]
    points = [[531, 230], [45, 321], [226, 360]]
    seg = Segmenter('models/FastSAM-x.pt')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg.prompt = frame
    seg.get_mask_by_box_prompt(bboxes)
    mask, unique_mask = seg.convert_mask_to_color()
    seg.clear_memory()
    masks = []
    images = []
    traker = TrackerCore()
    frames_to_propagate = 200
    current_frame_index = 0
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        current_memory_usage = psutil.virtual_memory().percent
        # print(current_memory_usage)
        if current_memory_usage > 90:
            break
        ret, frame_v = cap.read()
        if not ret:
            break
        if current_frame_index > frames_to_propagate:
            break

        if current_frame_index == 0:
            mask = traker.track(frame_v, unique_mask)
            masks.append(mask)
            images.append(frame_v)
        else:
            mask = traker.track(frame_v)
            masks.append(mask)
            images.append(frame_v)

        current_frame_index += 1
    video.release()
    print(len(masks))
    im3 = visualize_wb_mask(masks[200])
    ima = images[200].copy()
    for m in mask_map(masks[200]):
        for box in getting_coordinates(m):
            (x, y, w, h) = [v for v in box]
            cv2.rectangle(ima, (x, y), (x + w, y + h), (0, 255, 0), 2)
    image_m = overlay_davis(images[200], masks[200])
    cv2.imshow('image_m200', image_m)
    cv2.imshow('ima_rect', ima)
    cv2.imshow('im2', im3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
