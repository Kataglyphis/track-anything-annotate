import cv2
import numpy as np
import psutil
from tqdm import tqdm

from tools.mask_merge import merge_masks
from tracker_core import TrackerCore
from tools.overlay_image import painter_borders
from XMem.inference.interact.interactive_utils import overlay_davis
from sam_controller import SegmenterController
from interactive_video import InteractVideo


def get_frames(video_path: str, frames_to_propagate: int = 0):
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        current_frame_index = 0
        if frames_to_propagate == 0:
            frames_to_propagate = count_frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame_index > frames_to_propagate:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_frame_index += 1
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    return frames, fps


class Tracking:
    def __init__(self):
        self.sam_controller = SegmenterController()
        self.trecker = TrackerCore()

    def select_object(self, prompts: dict) -> np.ndarray:
        # maskss = []
        # for point in points:
        #     prompts = {
        #         'point_coords': np.array([point]),
        #         'point_labels': np.array([1]),
        #     }
        #     masks, scores, logits = self.segmenter.predict(prompts, 'point')
        #     maskss.append(masks[np.argmax(scores)])
        results = self.sam_controller.predict_from_prompts(prompts)
        results_masks = [
            result[np.argmax(scores)] for result, scores, logits in results
        ]
        mask, unique_mask = merge_masks(results_masks)
        return unique_mask

    def tracking(self, frames: list[np.ndarray], template_mask: np.ndarray) -> list:
        masks = []
        for i in tqdm(range(len(frames)), desc='Tracking'):
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 90:
                break
            """
             TODO: улучшение точности
                - надо проверять сколько масок в трекере
                - смотреть сколько объектов обнаруживается
                - если они не совпадают добавлять к новым маскам маску из трекера
            """
            if i == 0:
                mask = self.trecker.track(frames[i], template_mask)
                masks.append(mask)
            else:
                mask = self.trecker.track(frames[i])
                masks.append(mask)
        return masks

    def tracking_cut(
        self, frames: list[np.ndarray], templates_masks: dict[str, np.ndarray]
    ):
        masks = []
        exhaustive = False
        for i in tqdm(range(len(frames)), desc='Tracking_cut'):
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 90:
                break

            if str(i) in templates_masks:
                template_mask = templates_masks[str(i)]

            if i == 0 and str(i) in templates_masks:
                mask = self.trecker.track(frames[i], template_mask, exhaustive)
                masks.append(mask)
            else:
                mask = self.trecker.track(frames[i])
                masks.append(mask)

            if len(templates_masks) > 1:
                exhaustive = True

        return masks


if __name__ == '__main__':
    path = 'video-test/VID_20241218_134328.mp4'
    key_interval = 30
    controller = InteractVideo(path, key_interval)
    controller.extract_frames(300)  # Сначала извлекаем все кадры
    controller.collect_keypoints()
    results = controller.get_results()
    tracking = Tracking()
    frames = results['frames']

    # prompts = {
    #     'mode': 'point',
    #     'point_coords': [[531, 230], [45, 321], [226, 360], [194, 313]],
    #     'point_labels': [1, 1, 1, 1],
    # }

    select_masks = {}
    for frame_idx, points in results['keypoints'].items():

        if len(points) != 0:
            tracking.sam_controller.load_image(frames[int(frame_idx)])
            prompts = {
                'mode': 'point',
                'point_coords': points,
                'point_labels': [1] * len(points),
            }
            mask = tracking.select_object(prompts)
            select_masks[frame_idx] = mask

            # f = overlay_davis(frames[frame_idx], mask)
            # cv2.imshow('asd', f)
            # cv2.waitKey(0)
            tracking.sam_controller.reset_image()

    # masks = tracking.tracking(frames, mask)

    if len(select_masks) > 1:
        masks = []
        curr_frame = 0
        next = key_interval - 1
        frames_idx = list(select_masks.keys())
        mask = list(select_masks.values())
        for frame_idx, mask in select_masks.items():
            next += int(frame_idx)
            m = tracking.tracking_cut(frames[curr_frame:next], mask)
            masks += m
            tracking.trecker.clear_memory()
            curr_frame = next + 1

    filename = 'output_video_from_file_mem2_ved.mp4'
    output = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*'XVID'), controller.fps, controller.frame_size
    )
    for frame, mask in zip(frames, masks):
        # f = painter_borders(frame, mask)
        f = overlay_davis(frame, mask)
        output.write(f)
    # Освобождаем ресурсы
    output.release()
    cv2.destroyAllWindows()

    print(f'Видео записано в файл: {filename}')
