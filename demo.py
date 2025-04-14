import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw
from XMem2.inference.interact.interactive_utils import overlay_davis
from sam_controller import SegmenterController
from tracker import Tracker
from tracker_core_xmem2 import TrackerCore


# --- Извлечение всех кадров ---
def extract_all_frames(video_input):
    video_path = video_input
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))

    tracker.sam_controller.load_image(frames[0])
    video_state = {
        "fps": fps,
        "count_frames": count_frames,
    }
    return frames[0], frames, video_state


# --- Ручная разметка точками (первый кадр) ---
def on_image_click(image, evt: gr.SelectData, annotations_state):
    x, y = evt.index[0], evt.index[1]
    annotations_state["point"].append([x, y])

    # Отрисовка всех точек
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for ann in annotations_state["point"]:
        x_p, y_p = ann
        draw.ellipse((x_p - 5, y_p - 5, x_p + 5, y_p + 5), fill="blue")
    return img, annotations_state, f"Точка добавлена: ({x}, {y})"


# --- Разметка всех кадров ---
def tracking(frames: np.ndarray, video_state: dict) -> list[np.ndarray]:
    tracker.sam_controller.reset_image()
    masks = tracker.tracking(frames, video_state["mask"])
    video_state["annotations_masks"] = masks
    video_state["annotation_images"] = [
        overlay_davis(frame, mask) for frame, mask in zip(frames, masks)
    ]
    tracker.tracker.clear_memory()
    return video_state, video_state["annotation_images"]


# --- Аннотация ---
def annotations(
    frame: np.ndarray, annotations_state: dict, video_state: dict
) -> list[np.ndarray]:
    prompts = {
        'mode': 'point',
        'point_coords': annotations_state["point"],
        'point_labels': [1] * len(annotations_state["point"]),
    }
    mask = tracker.select_object(prompts)
    image = overlay_davis(frame, mask)
    video_state["mask"] = mask
    return image, video_state


segmenter_controller = SegmenterController()
tracker_core = TrackerCore()
tracker = Tracker(segmenter_controller, tracker_core)

# --- Интерфейс Gradio ---
with gr.Blocks() as demo:

    # Состояния
    frames = gr.State([])
    video_state = gr.State(
        {
            "fps": 30,
            "count_frames": 0,
            "mask": None,
            "annotations_masks": [],
            "annotation_images": [],
        }
    )
    annotations_state = gr.State({"frame_id": 0, "point": []})

    gr.Markdown("# Разметка видео: точки + боксы")

    with gr.Row():
        video_input = gr.Video(label="Загрузите видео")
        output_text = gr.Textbox(label="Результат")

    with gr.Row():
        annotations_btn = gr.Button("Аннотация")
        tracking_btn = gr.Button("Трекинг")

    with gr.Row():
        first_frame = gr.Image(label="Первый кадр (ручная разметка)", interactive=True)
        annotated_gallery = gr.Gallery(label="Все кадры с разметкой", columns=2)

    video_input.change(
        extract_all_frames,
        inputs=video_input,
        outputs=[first_frame, frames, video_state],
    )

    # Обработка кликов
    first_frame.select(
        on_image_click,
        inputs=[first_frame, annotations_state],
        outputs=[first_frame, annotations_state, output_text],
    )

    annotations_btn.click(
        annotations,
        inputs=[first_frame, annotations_state, video_state],
        outputs=[first_frame, video_state],
    )

    tracking_btn.click(
        tracking,
        inputs=[frames, video_state],
        outputs=[video_state, annotated_gallery],
    )

demo.launch(debug=True, server_port=8080)
