import numpy as np
import torch
from XMem.inference.inference_core import InferenceCore
from XMem.model.network import XMem
from config import XMEM_CONFIG, DEVICE
from torchvision import transforms
from XMem.util.range_transform import im_normalization

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

    def track(self, frame: np.ndarray, mask: np.ndarray):
        num_objects = len(np.unique(mask)) - 1
        self.processor.set_all_labels(range(1, num_objects + 1))

        frame_tensor = self.im_transform(frame).to(self.device)
        probs = self.processor.step(frame_tensor, mask, num_objects + 1)

        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

    @torch.no_grad()
    def clear_memory(self):
        self.processor.clear_memory()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    pass