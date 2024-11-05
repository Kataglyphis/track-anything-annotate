import cv2
import numpy as np

from fastsam import FastSAM, FastSAMPrompt, FastSAMPredictor
from fastsam.utils import convert_box_xywh_to_xyxy


class Segmenter:
    def __init__(self, model_path: str, device: str = 'cuda') -> None:
        self.device = device
        self.model = FastSAM(model_path)
        overrides = self.model.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(
            device=self.device, retina_masks=True, imgsz=1024, conf=0.7, iou=0.9
        )
        overrides['mode'] = 'predict'
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = False
        self.model.predictor = FastSAMPredictor(overrides=overrides)
        self.model.predictor.setup_model(model=self.model.model, verbose=False)

    @property
    def prompt(self):
        return self.prompt_process

    @prompt.setter
    def prompt(self, image: np.ndarray):
        everything_results = self.model.predictor(image)
        self.prompt_process = FastSAMPrompt(
            image, everything_results, device=self.device
        )

    def get_mask_by_box_prompt(self, bboxes: list[int]):
        box_prompt = [convert_box_xywh_to_xyxy(box) for box in bboxes]
        self.mask = self.prompt_process.box_prompt(bboxes=box_prompt)

    def convert_mask_to_color(self):
        mask = self.mask
        mask = np.uint8(mask) * 255
        mask_end = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        colors = [
            tuple(list(np.random.randint(0, 255, size=3))) for _ in range(len(mask))
        ]

        for i in range(len(mask)):
            color_mask = np.zeros_like(mask_end)
            color_mask[mask[i] > 0] = colors[i]
            mask_end = cv2.addWeighted(mask_end, 1, color_mask, 1, 0.0)

        return mask_end

    def convert_mask_to_white(self):
        mask = self.mask
        mask = np.uint8(mask) * 255
        mask_end = mask[0]
        if len(mask) == 1:
            return mask_end

        for i in range(1, len(mask)):
            mask_end = cv2.addWeighted(mask_end, 1, mask[i], 1, 0.0)

        return mask_end

    def annotated_frame(self) -> np.ndarray:
        annotated_frame = self.prompt_process.plot_to_result(
            annotations=self.mask,
            withContours=False,
            better_quality=True,
        )
        return annotated_frame


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()
    bboxes = []
    for _ in range(2):
        bbox = cv2.selectROI(frame_cop)
        bboxes.append(bbox)
    seg = Segmenter('models/FastSAM-x.pt')
    seg.prompt = frame
    seg.get_mask_by_box_prompt(bboxes)
    mask = seg.convert_mask_to_color()
    im = seg.annotated_frame()
    cv2.imshow('mask', mask)
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
