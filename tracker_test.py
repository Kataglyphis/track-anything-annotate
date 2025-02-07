import cv2
import numpy as np
import psutil
from tqdm import tqdm

from segmenter import Segmenter
from tracker_core_test import TrackerCore
from tools.overlay_image import painter_borders
from XMem2.inference.interact.interactive_utils import overlay_davis


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
        self.segmenter = Segmenter('models/FastSAM-x.pt')
        self.trecker = TrackerCore()

    def select_object(self, frame: np.ndarray) -> np.ndarray:
        bboxes = [(476, 166, 102, 154), (8, 252, 91, 149), (106, 335, 211, 90)]
        points = [[531, 230], [45, 321], [226, 360]]
        self.segmenter.prompt = frame
        self.segmenter.get_mask_by_box_prompt(bboxes)
        mask, unique_mask = self.segmenter.convert_mask_to_color()
        return unique_mask

    def tracking(self, frames: list, template_mask: np.ndarray) -> list:
        masks = []
        for i in tqdm(range(len(frames)), desc='Tracking'):
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 90:
                break

            if i == 0:
                mask = self.trecker.track(frames[i], template_mask)
                masks.append(mask)
            else:
                mask = self.trecker.track(frames[i])
                masks.append(mask)
        return masks


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    frame_size = (frame_width, frame_height)
    video.release()
    frames, fps = get_frames(path)
    tracking = Tracking()
    mask = tracking.select_object(frame)
    masks = tracking.tracking(frames, mask)
    filename = 'output_video_from_file_mem2.mp4'
    output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    for frame, mask in zip(frames, masks):
        # f = painter_borders(frame, mask)
        f = overlay_davis(frame, mask)
        output.write(f)
    # Освобождаем ресурсы
    output.release()
    cv2.destroyAllWindows()

    print(f'Видео записано в файл: {filename}')
