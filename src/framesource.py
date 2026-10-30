"""Read four camera videos as batches of aligned frame indices."""

from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent  # src/ is one level below the project root
VIDEO_DIR = ROOT / "data" / "wildtrack"
VIDEOS = {f"C{i}": VIDEO_DIR / f"cam{i}.mp4" for i in range(1, 5)}


def iter_frames(videos=VIDEOS, sample_fps=5):
    """Yield (original_frame_index, {camera_id: BGR_image}).

    Matching frame indices assumes aligned recording starts, not exact synchrony.
    """
    print(f"Reading videos from: {VIDEO_DIR}")
    captures = {}
    try:
        for camera, path in videos.items():
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {camera} video: {path}\n"
                    f"Expected recordings in: {VIDEO_DIR}\n"
                    "Check the filenames (cam1.mp4 ... cam4.mp4) and extension."
                )
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                raise OSError(f"Cannot open {camera} video: {path}")
            captures[camera] = cap

        rates = {camera: cap.get(cv2.CAP_PROP_FPS)
                 for camera, cap in captures.items()}
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if any(rate <= 0 for rate in rates.values()):
            raise ValueError(f"Video frame rate unavailable: {rates}")
        if max(rates.values()) - min(rates.values()) > 0.5:
            raise ValueError(f"Camera frame rates differ: {rates}")

        stride = max(1, round(next(iter(rates.values())) / sample_fps))
        print(f"Input FPS: {rates}; decoding every {stride}th frame")

        frame_index = 0
        while True:
            # Advance all streams; only decode the frames selected for inference.
            if not all(cap.grab() for cap in captures.values()):
                return

            if frame_index % stride == 0:
                batch = {}
                for camera, cap in captures.items():
                    ok, image = cap.retrieve()
                    if not ok:
                        raise OSError(f"Cannot decode {camera} frame {frame_index}")
                    batch[camera] = image
                yield frame_index, batch

            frame_index += 1
    finally:
        for cap in captures.values():
            cap.release()


if __name__ == "__main__":
    reader = iter_frames()
    try:
        for i, (frame_id, frames) in enumerate(reader):
            shapes = ", ".join(f"{camera}={image.shape}"
                               for camera, image in frames.items())
            print(f"Frame {frame_id}: {shapes}")
            if i == 2:
                break
    finally:
        reader.close()
