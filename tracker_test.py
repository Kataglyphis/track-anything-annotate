import cv2
import numpy as np
import psutil
from tqdm import tqdm

from tools.mask_merge import merge_masks
from tracker_core_test import TrackerCore
from tools.overlay_image import painter_borders
from XMem2.inference.interact.interactive_utils import overlay_davis
from sam_controller import SegmenterController
from interactive_video import InteractVideo


class Tracking:
    def __init__(self):
        self.sam_controller = SegmenterController()
        self.tracker = TrackerCore()

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

    def tracking(
        self,
        frames: list[np.ndarray],
        template_mask: np.ndarray,
        exhaustive: bool = False,
    ) -> list:
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
                mask = self.tracker.track(frames[i], template_mask, exhaustive)
                masks.append(mask)
            else:
                mask = self.tracker.track(frames[i])
                masks.append(mask)
        return masks

    def tracking_cut(
        self,
        frames: list[np.ndarray],
        templates_masks: dict[str, np.ndarray],
        exhaustive: bool = False,
    ):
        masks = []
        for i in tqdm(range(len(frames)), desc='Tracking_cut'):
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 90:
                break

            if str(i) in templates_masks:
                template_mask = templates_masks[str(i)]

            if i == 0 and str(i) in templates_masks:
                mask = self.tracker.track(frames[i], template_mask, exhaustive)
                masks.append(mask)
            else:
                mask = self.tracker.track(frames[i])
                masks.append(mask)

            if len(templates_masks) > 1:
                exhaustive = True

        return masks


if __name__ == '__main__':
    path = 'video-test/VID_20241218_134328.mp4'
    key_interval = 3
    controller = InteractVideo(path, key_interval)
    controller.extract_frames()  # Сначала извлекаем все кадры
    controller.collect_keypoints()
    results = controller.get_results()
    tracking = Tracking()
    frames = results['frames']

    # prompts = {
    #     'mode': 'point',
    #     'point_coords': [[531, 230], [45, 321], [226, 360], [194, 313]],
    #     'point_labels': [1, 1, 1, 1],
    # }

    frames_idx = list(map(int, results['keypoints'].keys()))
    frames_idx.append(len(controller.frames))
    print(frames_idx)
    result = []
    for i in range(len(frames_idx) - 1):
        current_frame = frames_idx[i]
        current_coords = results['keypoints'][str(current_frame)]

        next_frame = frames_idx[i + 1]
        print(current_frame, next_frame)
        if current_coords:
            tracking.sam_controller.load_image(frames[current_frame])
            prompts = {
                'mode': 'point',
                'point_coords': current_coords,
                'point_labels': [1] * len(current_coords),
            }
            mask = tracking.select_object(prompts)
            tracking.sam_controller.reset_image()
            result.append(
                {
                    "gap": [current_frame, next_frame],
                    "frame": current_frame,
                    "mask": mask,
                }
            )
        else:
            result.append(
                {
                    "gap": [current_frame, next_frame],
                    "frame": current_frame,
                    "mask": None,
                }
            )

    # select_masks = {}
    # points_frames = []
    # for frame_idx, points in results['keypoints'].items():
    #     if points:
    #         tracking.sam_controller.load_image(frames[int(frame_idx)])
    #         prompts = {
    #             'mode': 'point',
    #             'point_coords': points,
    #             'point_labels': [1] * len(points),
    #         }
    #         mask = tracking.select_object(prompts)
    #         select_masks[frame_idx] = mask
    #         # f = overlay_davis(frames[frame_idx], mask)
    #         # cv2.imshow('asd', f)
    #         # cv2.waitKey(0)
    #         tracking.sam_controller.reset_image()
    #     points_frames.append(int(frame_idx))

    # masks = tracking.tracking(frames, mask)
    print(result)
    masks = []
    for res in result:
        current_frame, next_frame = res['gap']
        if res['mask'] is not None:
            print(current_frame, next_frame)
            mask = tracking.tracking(frames[current_frame:next_frame], res['mask'])
            tracking.tracker.clear_memory()
            masks += mask
        else:
            print(current_frame, next_frame)
            m = []
            for _ in range(current_frame, next_frame):
                height, width, _ = frames[current_frame].shape
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                binary_mask[:, :] = 1
                m.append(binary_mask)
            masks += m

    # if len(select_masks) > 1:
    #     masks = []
    #     curr_frame = 0
    #     end_frame = key_interval
    #     for frame_idx, mask in select_masks.items():
    #         print(f'{curr_frame=} {end_frame=} ')
    #         print(len(frames[curr_frame:end_frame]))
    #         m = tracking.tracking(frames[curr_frame:end_frame], mask)
    #         tracking.tracker.clear_memory()

    #         masks += m
    #         curr_frame = end_frame
    #         end_frame += key_interval

    filename = 'output_video_from_file_mem2_ved_pot.mp4'
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
