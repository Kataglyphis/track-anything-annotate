import cv2
import uuid
import shutil
import numpy as np
from pathlib import Path
from typing import Protocol, Type

from tools.contour_detector import getting_coordinates
from tools.mask_display import mask_map


SAVE_FOLDER = Path.cwd() / 'video-test'


class ExportObject:
    def __init__(self, mask, name_class) -> None:
        self.mask = mask
        self.name_class = name_class


class ExportImage:
    def __init__(self, image, objects: list[ExportObject]) -> None:
        self.image = image
        self.exports_objects = objects


class TypeSave(Protocol):
    def start_creation(self) -> None:
        pass

    def create_archive(self) -> str:
        pass


def get_type(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names_class: list[str],
    type_save: str = 'yolo',
) -> TypeSave:
    '''Factory'''
    types_saves: dict[str, Type[TypeSave]] = {
        'yolo': YoloSave,
    }

    return types_saves[type_save](images, masks, names_class)


class YoloSave:
    def __init__(
        self, images: list[np.ndarray], masks: list[np.ndarray], names_class: list[str]
    ) -> None:
        self.images = images
        self.masks = masks
        self.names_class = {}

        for i, name in enumerate(names_class):
            self.names_class[name] = i
        folder_name = (
            ''.join(names_class)[:10]
            if len(''.join(names_class)) > 15
            else ''.join(names_class)
        )
        folder_name = f'dt-{folder_name}-{uuid.uuid1()}'
        p = Path(SAVE_FOLDER / folder_name)
        p.mkdir()
        self.path_folder = SAVE_FOLDER / folder_name
        self.images_folder = self.path_folder / 'images'
        p = Path(self.images_folder)
        p.mkdir()
        self.lables_folder = self.path_folder / 'lables'
        p = Path(self.lables_folder)
        p.mkdir()

    def start_creation(self):
        path_image = self.images_folder / 'image_filename'
        path_txt = self.lables_folder / 'image_filename'

        for i, (image, mask) in enumerate(zip(self.images, self.masks)):
            cv2.imwrite(f'{path_image}{i+1}.jpg', image)
            txt_frame_save(image, mask, f'{path_txt}{i+1}', 0)

        txt_class_save(self.path_folder / 'classes', self.names_class)

    def create_archive(self) -> str:
        shutil.make_archive(self.path_folder, 'zip', self.path_folder)
        shutil.rmtree(self.path_folder)
        return f'{self.path_folder}.zip'


def txt_class_save(path: str, names_class: dict):
    with open(f'{path}.txt', 'w') as file:
        for key, value in names_class.items():
            name_class_str = [f'{value} {key} \n']

            file.writelines(name_class_str)


def txt_frame_save(
    images: np.ndarray, mask_unique: np.ndarray, path: str, name_class_idx: int
):

    img_height = images.shape[0]
    img_width = images.shape[1]
    with open(f'{path}.txt', 'w') as file:
        coordinates = []
        for mask in mask_map(mask_unique):
            coordinate = getting_coordinates(mask)
            coordinates += coordinate

        for box in coordinates:
            x, y = box[0], box[1]
            w, h = box[2], box[3]

            x_center = x + int(w / 2)
            y_center = y + int(h / 2)

            norm_xc = x_center / img_width
            norm_yc = y_center / img_height
            norm_width = w / img_width
            norm_height = h / img_height

            yolo_annotation = [
                f'{name_class_idx} {norm_xc} {norm_yc} {norm_width} {norm_height} \n'
            ]

            file.writelines(yolo_annotation)
