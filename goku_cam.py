"""VanishCam Goku mode for macOS: Instant Transmission on your webcam.

Hold index + middle finger together against your forehead for 1 second to
teleport away; lower your hand and repeat to come back. The output goes to the
OBS Virtual Camera (and to a preview window you can also Window-Capture in OBS).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from goku_effect import (
    HOLD_SECONDS, IN_SECONDS, OUT_SECONDS, VANISH_AT, FaceBox, GestureHold,
    SpeedLines, goku_pose, render_teleport,
)
from teleport_sound import TeleportSound


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Anteprima"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
FACE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
MODEL_URLS = {
    SEGMENTER_MODEL: "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                     "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_detector/"
                "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task",
}

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
FACE_MEMORY_SECONDS = 1.5  # the hand on the forehead can hide the face briefly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Goku Instant Transmission webcam (macOS).")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--fps", type=float, default=30.0, help="Virtual camera FPS")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require OBS Virtual Camera (default: use it if available)",
    )
    parser.add_argument("--hold", type=float, default=HOLD_SECONDS,
                        help="Seconds to hold the gesture (default: 1)")
    parser.add_argument("--gesture-debug", action="store_true",
                        help="Print recognition details to the terminal")
    parser.add_argument("--self-test", action="store_true",
                        help="Load models and sound without opening the camera")
    return parser.parse_args()


def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if destination.is_file() and destination.stat().st_size > 100_000:
            continue
        print(f"[Goku] Downloading {destination.name}...")
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "VanishCam-Goku/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1 << 18):
                out.write(chunk)
        temporary.replace(destination)


def create_tasks(stack: ExitStack):
    video = vision.RunningMode.VIDEO
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=video, output_confidence_masks=True, output_category_mask=False,
    ))
    stack.callback(segmenter.close)
    faces = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=video, min_detection_confidence=0.45,
    ))
    stack.callback(faces.close)
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=video, num_hands=2,
        min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.45,
    ))
    stack.callback(hands.close)
    return segmenter, faces, hands


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} unavailable. Allow camera access for your terminal in "
            "System Settings > Privacy & Security > Camera, or try --camera 1."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(60):  # the first frames can take a moment on macOS
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            return capture
        time.sleep(0.05)
    capture.release()
    raise RuntimeError(
        "The webcam opened but sent no frames. Check camera permission for your "
        "terminal (System Settings > Privacy & Security > Camera) and close other apps using it."
    )


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(width=width, height=height, fps=args.fps,
                                     fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "OBS Virtual Camera unavailable. Install OBS Studio 30+, start its virtual "
            "camera once so macOS activates the extension, then STOP it in OBS before "
            f"running this app. Detail: {error}"
        )
        if args.virtual_camera:
            raise RuntimeError(message) from error
        print(f"[Goku] {message}\n[Goku] Continuing with the preview window only.")
        return None
    print(f"[Goku] Virtual output: {camera.device} ({width}x{height} @ {args.fps:g} fps)")
    return camera


class Timestamps:
    """MediaPipe VIDEO tasks need strictly increasing timestamps."""

    def __init__(self):
        self.started = time.perf_counter()
        self.last = -1

    def next(self) -> int:
        self.last = max(self.last + 1, int((time.perf_counter() - self.started) * 1000))
        return self.last


def person_mask(segmenter, image: mp.Image, timestamp: int, width: int, height: int) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp)
    if not result.confidence_masks:
        return np.zeros((height, width), np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    mask = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.2)
    return np.clip((mask - 0.12) / 0.68, 0.0, 1.0)


def detect_face(detector, image: mp.Image, timestamp: int) -> FaceBox | None:
    result = detector.detect_for_video(image, timestamp)
    if not result.detections:
        return None
    detection = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    box = detection.bounding_box
    eye_y = None
    if len(detection.keypoints) >= 2:  # 0 = right eye, 1 = left eye (normalized)
        eye_y = (detection.keypoints[0].y + detection.keypoints[1].y) / 2 * image.height
    return FaceBox(box.origin_x, box.origin_y, box.origin_x + box.width,
                   box.origin_y + box.height, eye_y)


def detect_hands(landmarker, image: mp.Image, timestamp: int) -> list[np.ndarray]:
    result = landmarker.detect_for_video(image, timestamp)
    return [np.array([(p.x * image.width, p.y * image.height) for p in hand], np.float32)
            for hand in result.hand_landmarks]


def set_fullscreen(enabled: bool) -> None:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)


def self_test() -> int:
    download_models()
    with ExitStack() as stack:
        segmenter, faces, hands = create_tasks(stack)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((360, 640, 3), np.uint8))
        clock = Timestamps()
        person_mask(segmenter, image, clock.next(), 640, 360)
        detect_face(faces, image, clock.next())
        detect_hands(hands, image, clock.next())
    sound = TeleportSound()
    print(f"[Goku] Self-test OK: models load, sound = {sound.path.name}")
    return 0


def run(args: argparse.Namespace) -> int:
    download_models()
    with ExitStack() as stack:
        segmenter, faces, hands = create_tasks(stack)
        capture = open_camera(args.camera, args.width, args.height)
        stack.callback(capture.release)
        sound = TeleportSound()
        stack.callback(sound.close)
        virtual_camera = open_virtual_camera(args, args.width, args.height)
        if virtual_camera is not None:
            stack.callback(virtual_camera.close)
        stack.callback(cv2.destroyAllWindows)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
        fullscreen = args.fullscreen
        if fullscreen:
            set_fullscreen(True)

        clock = Timestamps()
        hold = GestureHold(hold_seconds=args.hold)
        phase = "calibrating"
        phase_started = time.perf_counter()
        background: np.ndarray | None = None
        samples: list[np.ndarray] = []
        countdown_shown = None
        face: FaceBox | None = None
        face_seen = float("-inf")
        mask: np.ndarray | None = None
        lines = SpeedLines()
        effects = 0
        last_report = 0.0
        last_hold_print = None
        print("[Goku] Step out of the frame for 4 seconds: capturing the background.")
        print("[Goku] Keys in the preview window: SPACE new background, F fullscreen, Q/ESC quit.")

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("The webcam stopped sending frames.")
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.perf_counter()
            image = None

            def mp_image() -> mp.Image:
                nonlocal image
                if image is None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                return image

            if phase == "calibrating":
                elapsed = now - phase_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != countdown_shown:
                    print(f"[Goku] Background capture: {remaining} s")
                    countdown_shown = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    samples.append(frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    background = np.median(np.stack(samples[-18:]), axis=0).astype(np.uint8)
                    samples.clear()
                    phase = "present"
                    hold = GestureHold(hold_seconds=args.hold)
                    print("[Goku] Background ready. Step back in, then put two fingers on "
                          "your forehead for 1 second to teleport.")
                display = frame

            else:
                # Gestures are read from the real camera, even while you are "gone".
                if phase in ("present", "vanished"):
                    found = face_now = detect_face(faces, mp_image(), clock.next())
                    if face_now is not None:
                        face, face_seen = face_now, now
                    elif now - face_seen > FACE_MEMORY_SECONDS:
                        face = None
                    hand_points = detect_hands(hands, mp_image(), clock.next())
                    posed = goku_pose(hand_points, face)
                    fired = hold.update(posed, now)
                    left = hold.remaining(now)
                    shown = None if left is None else int(np.ceil(left * 10))
                    if shown != last_hold_print and shown is not None and not fired:
                        print(f"[Goku] Hold... {left:.1f} s")
                    last_hold_print = shown
                    if args.gesture_debug and now - last_report >= 0.5:
                        print(f"[Debug] face={'yes' if found else ('memory' if face else 'no')} "
                              f"hands={len(hand_points)} pose={posed} armed={hold.armed}")
                        last_report = now
                    if fired:
                        effects += 1
                        lines = SpeedLines(seed=effects)
                        phase = "out" if phase == "present" else "in"
                        phase_started = now
                        sound.play()
                        print("[Goku] Instant Transmission!" if phase == "out" else "[Goku] Back!")
                        if phase == "in":
                            print("[Goku] Lower your hand before teleporting again.")

                if phase == "present":
                    display = frame
                elif phase == "vanished":
                    display = background
                elif phase == "out":
                    elapsed = now - phase_started
                    t = min(1.0, elapsed / OUT_SECONDS)
                    if t < VANISH_AT or mask is None:  # freeze the silhouette once gone
                        mask = person_mask(segmenter, mp_image(), clock.next(), width, height)
                    display = render_teleport(frame, background, mask, t, lines, elapsed)
                    if t >= 1.0:
                        phase, mask = "vanished", None
                        print("[Goku] Gone. Lower your hand, then repeat the gesture to come back.")
                else:  # "in": the reverse animation, with the live silhouette
                    elapsed = now - phase_started
                    t = max(0.0, 1.0 - elapsed / IN_SECONDS)
                    mask = person_mask(segmenter, mp_image(), clock.next(), width, height)
                    display = render_teleport(frame, background, mask, t, lines, elapsed)
                    if t <= 0.0:
                        phase, mask = "present", None

            if virtual_camera is not None:
                virtual_camera.send(np.ascontiguousarray(display))
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                phase, phase_started, countdown_shown, mask = "calibrating", now, None, None
                samples.clear()
                print("[Goku] Step out of the frame for 4 seconds: capturing a new background.")
            elif key == ord("f"):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    return 0


def main() -> int:
    args = parse_args()
    try:
        return self_test() if args.self_test else run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[Goku] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
