import cv2
import progressbar


class InteractVideo:
    def __init__(
        self, video_path: str, keyframe_interval: int = 3, one_frame: bool = False
    ):
        self.video_path = video_path
        self.frames = []
        self.keypoints = {}  # {frame_index: [(x1,y1), (x2,y2), ...]}
        self.keyframe_interval = keyframe_interval
        self.current_frame_idx = 0  # Текущий индекс кадра
        self.history = []  # Для отслеживания пропущенных кадров

    def extract_frames(
        self,
        frames_to_propagate: int = 0,
        max_width: int = 1280,
        max_height: int = 720,
    ):
        """Извлекает все кадры из видео и сохраняет в self.frames"""
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(3))
        original_height = int(cap.get(4))
        self.frame_size = (original_width, original_height)
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
            # Проверка и изменение размера кадра
            if max_width or max_height:
                h, w = frame.shape[:2]
                ratio_w = max_width / w if max_width else float('inf')
                ratio_h = max_height / h if max_height else float('inf')
                ratio = min(ratio_w, ratio_h, 1.0)  # Не увеличиваем изображение

                if ratio < 1.0:
                    new_size = (int(w * ratio), int(h * ratio))
                    frame = cv2.resize(frame, new_size)
                    # Обновляем размер кадра для первого кадра
                    if frame_index == 0:
                        self.frame_size = new_size
            self.frames.append(frame)
            frame_index += 1
            bar.update(frame_index)
        bar.finish()
        cap.release()

    def collect_keypoints(self):
        """Собирает ключевые точки с поддержкой навигации"""
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.mouse_callback)
        mode = ''
        while self.current_frame_idx < len(self.frames):
            frame = self.frames[self.current_frame_idx]
            is_keyframe = self.current_frame_idx % self.keyframe_interval == 0

            if is_keyframe:
                self.current_points = self.keypoints.get(
                    self.current_frame_idx, []
                ).copy()
                self.show_frame_with_controls(mode)

                while True:
                    key = cv2.waitKey(100)

                    # Подтверждение выбора
                    if key == 13 or key == ord('s'):  # Enter
                        self.keypoints[str(self.current_frame_idx)] = (
                            self.current_points.copy()
                        )
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        mode = 'Saver'
                        break
                    elif key == ord('w'):
                        self.keypoints[str(self.current_frame_idx)] = []
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        mode = 'Empty'
                        break
                    # Пропуск кадра
                    elif key == ord('d'):
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        break
                    # Назад
                    elif key == ord('a') and self.history:
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

    def show_frame_with_controls(self, mode):
        """Показывает кадр с элементами управления"""
        self.current_frame = self.frames[self.current_frame_idx].copy()
        h, w = self.current_frame.shape[:2]

        # Панель управления
        cv2.rectangle(self.current_frame, (0, 0), (w, 43), (0, 0, 0), -1)
        cv2.putText(
            self.current_frame,
            f"Frame {self.current_frame_idx} from {len(self.frames)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.current_frame,
            "Enter/s - save frame(start keyframe) a - back  d - next w - start empty frame(gap) q - quit",
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            self.current_frame,
            f"Current mode: {mode}",
            (300, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Сетка
        cv2.line(self.current_frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(self.current_frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        # Точки
        for x, y in self.current_points:
            cv2.circle(self.current_frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Frame", self.current_frame)

    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик кликов мыши"""
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and self.current_frame_idx % self.keyframe_interval == 0
        ):
            print(f'Кадр {self.current_frame_idx}')
            if len(self.current_points) < 10:
                self.current_points.append((x, y))
                print(f'Точка добавлена: ({x}, {y})')
                cv2.circle(self.current_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Frame", self.current_frame)
            else:
                print("Достигнут лимит точек (10)")

    def get_results(self):
        """Возвращает результаты с учётом пропущенных кадров"""
        return {
            'frames': self.frames,
            'keypoints': self.keypoints,
        }


if __name__ == '__main__':
    controller = InteractVideo('video-test/video.mp4')
    controller.extract_frames()  # Сначала извлекаем все кадры
    controller.collect_keypoints()
    results = controller.get_results()
    print(f'Всего кадров: {len(results["frames"])}')
    for frame_idx, points in results['keypoints'].items():
        if points:
            print(f"Кадр {frame_idx}: {len(points)} точек")
        else:
            print(f'Пустой кадр {frame_idx}')

    select_masks = {}
    points_frames = []
    for frame_idx, points in results['keypoints'].items():
        if points:
            select_masks[frame_idx] = len(points)
        points_frames.append(int(frame_idx))
    points_frames.append(len(controller.frames))

    print(f'{len(select_masks)=}')
    print(f'{select_masks=}')
    print(f'{len(points_frames)=}')
    print(f'{points_frames}')

    frames_idx = list(map(int, results['keypoints'].keys()))
    result = []
    for i in range(len(frames_idx) - 1):
        current_frame = frames_idx[i]
        current_coords = results['keypoints'][str(current_frame)]

        next_frame = frames_idx[i + 1]
        result.append(
            {
                "gap": [current_frame, next_frame],
                "frame": current_frame,
                "coords": current_coords if current_coords else None,
            }
        )

    print(result)
