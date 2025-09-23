import cv2
import uuid
import shutil
import numpy as np
import json
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
        'coco': CocoDatasetSevar,
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


class CocoDatasetSevar:
    def __init__(
        self, images: list[np.ndarray], masks: list[np.ndarray], class_names: list[str]
    ):
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

    def create_dataset(self):
        self._create_coco_annotations(self.images, self.masks)

    def create_archive(self):
        shutil.make_archive(self.dataset_dir, 'zip', self.dataset_dir)
        shutil.rmtree(self.dataset_dir)
        return f'{self.dataset_dir}.zip'

    def _create_coco_annotations(self, images: list, masks: list):
        coco_data = {
            # 'info': {
            #     'description': 'Custom COCO Dataset',
            #     'version': '1.0',
            #     'year': 2024,
            #     'contributor': '',
            #     'url': ''
            # },
            # 'licenses': [{'id': 1, 'name': 'Academic', 'url': ''}],
            'categories': self._create_categories(),
            'images': [],
            'annotations': [],
        }

        annotation_id = 1
        for img_id, (image, mask) in enumerate(zip(images, masks)):
            img_filename = f'{img_id:012d}.jpg'
            img_path = self.images_dir / img_filename
            cv2.imwrite(str(img_path), image)

            coco_data['images'].append(
                {
                    'id': img_id,
                    'file_name': img_filename,
                    'width': image.shape[1],
                    'height': image.shape[0],
                }
            )

            # Добавляем аннотации (bounding boxes и сегментации)
            annotations = self._create_annotations(mask, img_id, annotation_id)
            coco_data['annotations'].extend(annotations)
            annotation_id += len(annotations)

        # Сохраняем JSON аннотации
        annotations_path = self.dataset_dir / 'annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

    def _create_categories(self):
        return [
            {'id': class_id, 'name': class_name}
            for class_name, class_id in enumerate(self.class_to_idx)
        ]

    def _create_annotations(
        self, mask_unique: np.ndarray, image_id: int, start_id: int
    ):
        annotations = []
        coordinates = []
        for mask in mask_map(mask_unique):
            bbox = getting_coordinates(mask)
            coordinates += bbox
        for box in coordinates:
            x, y = box[0], box[1]
            w, h = box[2], box[3]
            data_images = {'image_id': image_id, 'category_id': 0, 'bbox': [x, y, w, h]}
            annotations.append(data_images)
        return annotations


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
