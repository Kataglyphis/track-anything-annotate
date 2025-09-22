import cv2
import uuid
import shutil
import numpy as np
from pathlib import Path
from typing import Protocol, Type

from tools.contour_detector import getting_coordinates
from tools.mask_display import mask_map


SAVE_FOLDER = Path.cwd() / 'video-test'


# class ExportObject:
#     def __init__(self, mask, name_class) -> None:
#         self.mask = mask
#         self.name_class = name_class


# class ExportImage:
#     def __init__(self, image, objects: list[ExportObject]) -> None:
#         self.image = image
#         self.exports_objects = objects


class TypeSave(Protocol):
    def create_dataset(self) -> None:
        pass

    def create_archive(self) -> str:
        pass


def get_type_save_annotation(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names_class: list[str],
    type_save: str = 'yolo',
) -> TypeSave:
    '''Factory'''
    types_saves: dict[str, Type[TypeSave]] = {
        'yolo': YoloDatasetSaver,
    }

    return types_saves[type_save](images, masks, names_class)


def generate_class_folder_name(names_class: list[str]):
    base_name = (
        ''.join(names_class)[:10]
        if len(''.join(names_class)) > 15
        else ''.join(names_class)
    )

    folder_name = f'dt-{base_name}-{uuid.uuid1()}'
    return folder_name


class YoloDatasetSaver:

    def __init__(
        self, images: list[np.ndarray], masks: list[np.ndarray], class_names: list[str]
    ) -> None:
        self.images = images
        self.masks = masks
        self.class_to_idx = {}

        for i, name in enumerate(class_names):
            self.class_to_idx[name] = i

        dataset_name = generate_class_folder_name(class_names)
        dataset_path = Path(SAVE_FOLDER / dataset_name)
        dataset_path.mkdir()

        self.dataset_dir = SAVE_FOLDER / dataset_name

        self.images_dir = self.dataset_dir / 'images'
        images_dir = Path(self.images_dir)
        images_dir.mkdir()

        self.labels_dir = self.dataset_dir / 'labels'
        labels_dir = Path(self.labels_dir)
        labels_dir.mkdir()

    def create_dataset(self):
        for idx, (image, mask) in enumerate(zip(self.images, self.masks)):
            image_filename = f'image_{idx+1:04d}'
            image_path = self.images_dir / f'{image_filename}.jpg'
            label_path = self.labels_dir / f'{image_filename}.txt'

            cv2.imwrite(str(image_path), image)
            self._save_yolo_annotation(image, mask, str(label_path), 0)

        self._save_class_names(self.dataset_dir / 'classes.txt')

    def create_archive(self) -> str:
        shutil.make_archive(self.dataset_dir, 'zip', self.dataset_dir)
        shutil.rmtree(self.dataset_dir)
        return f'{self.dataset_dir}.zip'

    def _save_class_names(self, file_path: str):
        with open(file_path, 'w') as file:
            for class_name, class_id in self.class_to_idx.items():
                file.write(f'{class_id} {class_name}\n')

    def _save_yolo_annotation(
        self,
        images: np.ndarray,
        mask_unique: np.ndarray,
        file_path: str,
        name_class_idx: int,
    ):

        img_height = images.shape[0]
        img_width = images.shape[1]
        with open(file_path, 'w') as file:
            coordinates = []
            for mask in mask_map(mask_unique):
                bbox = getting_coordinates(mask)
                coordinates += bbox

            for box in coordinates:
                x, y = box[0], box[1]
                w, h = box[2], box[3]

                x_center = x + w / 2
                y_center = y + h / 2

                norm_xc = x_center / img_width
                norm_yc = y_center / img_height
                norm_width = w / img_width
                norm_height = h / img_height

                file.write(
                    f'{name_class_idx} {norm_xc} {norm_yc} {norm_width} {norm_height}\n'
                )
