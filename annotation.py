from tools.data_exporter import get_type
from tracker_test import Tracking
from interactive_video import InteractVideo


def main(path: str, name_class: list[str]):
    controller_video = InteractVideo(path)
    controller_video.extract_frames()
    controller_video.collect_keypoints()
    results = controller_video.get_results()
    tracking = Tracking()
    frames = results['frames']
    frames_idx = list(map(int, results['keypoints'].keys()))

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

    masks = []
    images_ann = []
    for res in result:
        current_frame, next_frame = res['gap']
        if res['mask'] is not None:
            print(current_frame, next_frame)

            images = frames[current_frame:next_frame]
            mask = tracking.tracking(images, res['mask'])
            tracking.tracker.clear_memory()
            masks += mask
            images_ann += images

    assert len(masks) == len(images_ann)

    saves = get_type(images_ann, masks, name_class)
    saves.start_creation()
    saves.create_archive()


if __name__ == '__main__':
    path = 'video-test/VID_20241218_134328.mp4'
    name = ['tomato']
    main(path, name)
