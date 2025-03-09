import cv2
import numpy as np
import progressbar


class InteractVideo:
    def __init__(
        self, video_path: str, keyframe_interval: int = 40, one_frame: bool = False
    ):
        self.video_path = video_path
        self.frames = []
        self.keypoints = {}  # {frame_index: [(x1,y1), (x2,y2), ...]}
        self.keyframe_interval = keyframe_interval
        self.current_frame_idx = 0  # Текущий индекс кадра
        self.history = []  # Для отслеживания пропущенных кадров

    def extract_frames(self, frames_to_propagate: int = 0):
        """Извлекает все кадры из видео и сохраняет в self.frames"""
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        self.frame_size = (frame_width, frame_height)
        frame_index = 0
        count_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        print(f'Extracting frames from {self.video_path} into a temporary dir...')
        bar = progressbar.ProgressBar(max_value=int(count_frames))

        if frames_to_propagate == 0:
            frames_to_propagate = count_frames

        self.frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index > frames_to_propagate:
                break

            self.frames.append(frame)
            frame_index += 1
            bar.update(frame_index)
        bar.finish()
        cap.release()

    def collect_keypoints(self):
        """Собирает ключевые точки с поддержкой навигации"""
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.mouse_callback)

        while self.current_frame_idx < len(self.frames):
            frame = self.frames[self.current_frame_idx]
            is_keyframe = self.current_frame_idx % self.keyframe_interval == 0

            if is_keyframe:
                self.current_points = self.keypoints.get(
                    self.current_frame_idx, []
                ).copy()
                self.show_frame_with_controls()

                while True:
                    key = cv2.waitKey(100)

                    # Подтверждение выбора
                    if key == 13:  # Enter
                        self.keypoints[self.current_frame_idx] = (
                            self.current_points.copy()
                        )
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        break
                    # Пропуск кадра
                    elif key == ord('s'):
                        self.current_frame_idx += 1
                        break
                    # Назад
                    elif key == ord('z') and self.history:
                        prev_idx = self.history.pop()
                        self.current_frame_idx = prev_idx
                        break
                    # Выход
                    elif key in [ord('q'), 27]:
                        return

            else:
                # Показываем обычные кадры без остановки
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1)
                if key in [ord('q'), 27]:
                    break
                self.current_frame_idx += 1

        cv2.destroyAllWindows()

    def show_frame_with_controls(self):
        """Показывает кадр с элементами управления"""
        frame = self.frames[self.current_frame_idx].copy()
        h, w = frame.shape[:2]

        # Панель управления
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Кадр {self.current_frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Enter - save  Z - back  S - skip",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Сетка
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        # Точки
        for x, y in self.current_points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Frame", frame)

    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик кликов мыши"""
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and self.current_frame_idx % self.keyframe_interval == 0
        ):
            if len(self.current_points) < 10:
                self.current_points.append((x, y))
                print(f"Точка добавлена: ({x}, {y})")
            else:
                print("Достигнут лимит точек (10)")

    def get_results(self):
        """Возвращает результаты с учётом пропущенных кадров"""
        return {
            'frames': self.frames,
            'keypoints': self.keypoints,
        }


if __name__ == '__main__':
    controller = InteractVideo('video-test/VID_20241218_134328.mp4', 30)
    controller.extract_frames()  # Сначала извлекаем все кадры
    controller.collect_keypoints()
    results = controller.get_results()
    print(f'Всего кадров: {len(results['frames'])}')
    for frame_idx, points in results['keypoints'].items():
        print(f"Кадр {frame_idx}: {len(points)} точек")
