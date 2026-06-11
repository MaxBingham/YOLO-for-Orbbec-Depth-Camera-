
import cv2
import numpy as np
from ultralytics import YOLO

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBStreamType, AlignFilter, OBFormat
)

# Hilfsfunktion aus den pyorbbecsdk-Beispielen (examples/utils.py)
# Wandelt einen Orbbec-Farbframe robust in ein BGR-NumPy-Array um.
try:
    from utils import frame_to_bgr_image
except ImportError:
    def frame_to_bgr_image(color_frame):
        """Minimaler Fallback. Bei MJPG-Frames bitte die echte
        frame_to_bgr_image aus den SDK-Beispielen verwenden."""
        w = color_frame.get_width()
        h = color_frame.get_height()
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        fmt = color_frame.get_format()
        if fmt == OBFormat.RGB:
            img = data.reshape((h, w, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return data.reshape((h, w, 3))
        if fmt == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        raise ValueError(f"Unbehandeltes Farbformat: {fmt}")


def main():
    # --- YOLO-Pose laden (pretrained, kein Training) ---
    model = YOLO("yolo26n-pose.pt")

    # --- Orbbec-Pipeline einrichten ---
    pipeline = Pipeline()
    config = Config()

    # Farb- und Tiefenstream mit Default-Profil aktivieren
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    config.enable_stream(color_profiles.get_default_video_stream_profile())
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    config.enable_stream(depth_profiles.get_default_video_stream_profile())

    pipeline.start(config)

    # Tiefe auf Farbe ausrichten -> Tiefenpixel passen zu Keypoint-Pixeln.
    # WICHTIG: Nach dem Align fuer die Back-Projection die FARB-Intrinsics nutzen.
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    print("Streaming laeuft. 'q' oder ESC zum Beenden.")
    try:
        while True:
            frames = pipeline.wait_for_frames(100)  # ms
            if frames is None:
                continue

            frames = align.process(frames)
            if frames is None:
                continue
            frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            # Farbframe -> BGR (fuer YOLO und cv2.imshow)
            color = frame_to_bgr_image(color_frame)

            # Tiefenframe -> NumPy uint16 (Millimeter). Brauchst du gleich
            # fuer die Back-Projection: Z = depth[v, u] * depth_scale.
            dh, dw = depth_frame.get_height(), depth_frame.get_width()
            depth = np.frombuffer(depth_frame.get_data(),
                                  dtype=np.uint16).reshape((dh, dw))

            # --- YOLO-Pose (identisch zu deinem Webcam-Code) ---
            results = model(color, conf=0.25, verbose=False)
            annotated = results[0].plot()

            cv2.imshow("Orbbec + YOLO-Pose", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()