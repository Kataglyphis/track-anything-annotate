import numpy as np
from sam_controller import SegmenterController
from tools.data_exporter import get_type_save_annotation
from tracker import Tracker
from interactive_video import InteractVideo
from tracker_core_xmem2 import TrackerCore


def create_dataset(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names_class: list[str],
    type_save: str = 'yolo',
):
    assert len(masks) == len(images)

    send_images = []
    send_masks = []
    for i in range(len(masks)):
        if i % 2 == 0:
            send_images.append(images[i])
            send_masks.append(masks[i])

    saver = get_type_save_annotation(send_images, send_masks, names_class, type_save)
    saver.create_dataset()
    print(saver.create_archive())


def main(video_path: str, names_class: list[str]):
    video = InteractVideo(video_path)
    video.extract_frames()
    video.collect_keypoints()
    results = video.get_results()

    segmenter_controller = SegmenterController()
    tracker_core = TrackerCore()
    tracker = Tracker(segmenter_controller, tracker_core)

    annotations = []
    print(len(results['keypoints']))
    if len(results['keypoints']) == 1:
        current_frame = list(results['keypoints'].keys())[0]
        next_frame = len(results["frames"])
        current_coords = results['keypoints'][current_frame]

        if current_coords:
            tracker.sam_controller.load_image(results['frames'][int(current_frame)])
            prompts = {
                'mode': 'point',
                'point_coords': current_coords,
                'point_labels': [1] * len(current_coords),
            }
            mask = tracker.select_object(prompts)
            tracker.sam_controller.reset_image()
            annotations.append(
                {
                    'gap': [current_frame, next_frame],
                    'frame': current_frame,
                    'mask': mask,
                }
            )

    for i in range(len(results['keypoints']) - 1):
        current_frame = list(results['keypoints'].keys())[i]
        next_frame = list(results['keypoints'].keys())[i + 1]
        current_coords = results['keypoints'][current_frame]

        if current_coords:
            tracker.sam_controller.load_image(results['frames'][int(current_frame)])
            prompts = {
                'mode': 'point',
                'point_coords': current_coords,
                'point_labels': [1] * len(current_coords),
            }
            mask = tracker.select_object(prompts)
            tracker.sam_controller.reset_image()
            annotations.append(
                {
                    'gap': [current_frame, next_frame],
                    'frame': current_frame,
                    'mask': mask,
                }
            )

    print(f'{len(annotations)} Колличество сегментов')

    masks = []
    images_ann = []
    for ann in annotations:
        current_frame, next_frame = ann['gap']
        if ann['mask'] is not None:
            images = results['frames'][int(current_frame) : int(next_frame)]
            mask = tracker.tracking(images, ann['mask'])
            tracker.tracker.clear_memory()
            masks += mask
            images_ann += images

    create_dataset(images_ann, masks, names_class, 'coco')


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    name = ['tomato']
    main(path, name)
