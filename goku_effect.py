"""Goku "Instant Transmission": two-finger forehead gesture and teleport effect.

The gesture: index and middle fingers straight and held together against the
forehead, ring and little fingers folded. Holding it for ``HOLD_SECONDS`` makes
the person teleport away; lowering the hand and repeating it brings them back.

The effect is drawn in code: comic-style vertical speed lines run across the
silhouette, the person squeezes into a thin bright line and vanishes, and the
lines linger briefly over the empty background. The return plays it in reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


HOLD_SECONDS = 1.0
OUT_SECONDS = 1.15
IN_SECONDS = 1.0
# Timeline positions (0 = fully present, 1 = gone and lines faded).
SQUEEZE_START = 0.42
VANISH_AT = 0.56

WRIST = 0
INDEX, MIDDLE, RING, PINKY = 5, 9, 13, 17  # MCP landmark of each finger


# --------------------------------------------------------------------------- #
# Gesture                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class FaceBox:
    """Face position in pixels, from the BlazeFace detector."""

    x1: float
    y1: float
    x2: float
    y2: float
    eye_y: float | None = None

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


def _finger(points: np.ndarray, base: int) -> tuple[float, float, bool, bool]:
    """Return (length, straightness, extended, folded) for one finger."""
    finger = points[base : base + 4]
    length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
    if length < 1e-6:
        return 0.0, 0.0, False, True
    straightness = float(np.linalg.norm(finger[3] - finger[0])) / length
    tip_reach = float(np.linalg.norm(finger[3] - points[WRIST]))
    pip_reach = float(np.linalg.norm(finger[1] - points[WRIST]))
    extended = straightness >= 0.80 and tip_reach > pip_reach * 1.08
    folded = straightness < 0.75 or tip_reach <= pip_reach * 1.05
    return length, straightness, extended, folded


def two_fingers_together(points: np.ndarray) -> bool:
    """Index and middle extended side by side, ring and little finger folded."""
    if points.shape[0] != 21 or not np.isfinite(points).all():
        return False
    index_len, _, index_ext, _ = _finger(points, INDEX)
    middle_len, _, middle_ext, _ = _finger(points, MIDDLE)
    _, _, _, ring_folded = _finger(points, RING)
    _, _, _, pinky_folded = _finger(points, PINKY)
    if not (index_ext and middle_ext) or min(index_len, middle_len) < 1e-6:
        return False

    # Seen from the side, folded fingers can look straight but short.
    reference = min(index_len, middle_len)
    ring_folded = ring_folded or np.linalg.norm(points[RING + 3] - points[RING]) < reference * 0.65
    pinky_folded = pinky_folded or np.linalg.norm(points[PINKY + 3] - points[PINKY]) < reference * 0.65
    if not (ring_folded and pinky_folded):
        return False

    # Together, not a V: tips about a finger-width apart and pointing the same way.
    tip_gap = float(np.linalg.norm(points[INDEX + 3] - points[MIDDLE + 3]))
    index_dir = points[INDEX + 3] - points[INDEX]
    middle_dir = points[MIDDLE + 3] - points[MIDDLE]
    cosine = float(np.dot(index_dir, middle_dir) /
                   max(1e-6, np.linalg.norm(index_dir) * np.linalg.norm(middle_dir)))
    return tip_gap <= reference * 0.50 and cosine >= 0.88


def fingers_on_forehead(points: np.ndarray, face: FaceBox) -> bool:
    """Index/middle finger pads touch the forehead area above the eyes."""
    eye_y = face.eye_y if face.eye_y is not None else face.y1 + 0.40 * face.height
    left = face.x1 - 0.10 * face.width
    right = face.x2 + 0.10 * face.width
    top = face.y1 - 0.60 * face.height
    bottom = eye_y + 0.03 * face.height

    # PIP, DIP and tip of index and middle finger.
    contact = points[[6, 7, 8, 10, 11, 12], :2]
    inside = ((contact[:, 0] >= left) & (contact[:, 0] <= right)
              & (contact[:, 1] >= top) & (contact[:, 1] <= bottom))
    if np.count_nonzero(inside) < 2:
        return False
    # Ignore small hands far behind the user.
    finger_length = float(np.linalg.norm(points[8, :2] - points[5, :2]))
    return finger_length >= 0.22 * face.height


def goku_pose(hands: list[np.ndarray], face: FaceBox | None) -> bool:
    """``hands`` holds 21x2 (or 21x3) landmark arrays in pixel coordinates."""
    if face is None:
        return False
    return any(two_fingers_together(points[:, :2]) and fingers_on_forehead(points, face)
               for points in hands)


@dataclass
class GestureHold:
    """Fires once after the pose is held; re-arms only after the hand is lowered."""

    hold_seconds: float = HOLD_SECONDS
    release_seconds: float = 0.25
    grace_seconds: float = 0.15  # tolerate single-frame detector dropouts
    armed: bool = True
    started: float | None = None
    last_seen: float | None = None
    released_since: float | None = None

    def remaining(self, now: float) -> float | None:
        if self.started is None:
            return None
        return max(0.0, self.hold_seconds - (now - self.started))

    def update(self, active: bool, now: float) -> bool:
        if not self.armed:
            if active:
                self.released_since = None
            elif self.released_since is None:
                self.released_since = now
            elif now - self.released_since >= self.release_seconds:
                self.armed = True
                self.released_since = None
            return False

        if active:
            self.last_seen = now
            if self.started is None:
                self.started = now
            if now - self.started >= self.hold_seconds:
                self.disarm()
                return True
        elif self.started is not None and (self.last_seen is None or
                                           now - self.last_seen > self.grace_seconds):
            self.started = None  # pose broken: the countdown starts over
        return False

    def disarm(self) -> None:
        self.armed = False
        self.started = None
        self.last_seen = None
        self.released_since = None


# --------------------------------------------------------------------------- #
# Effect                                                                       #
# --------------------------------------------------------------------------- #

def smoothstep(edge0: float, edge1: float, value: float) -> float:
    value = min(1.0, max(0.0, (value - edge0) / max(edge1 - edge0, 1e-6)))
    return value * value * (3.0 - 2.0 * value)


def line_strength(t: float) -> float:
    """Speed-line density: builds up, peaks at the vanish, then lingers out."""
    if t < VANISH_AT:
        return smoothstep(0.0, 0.40, t)
    return 1.0 - smoothstep(VANISH_AT + 0.06, 1.0, t)


def squeeze_amount(t: float) -> float:
    """Horizontal scale of the person: 1 = normal, 0 = gone."""
    return 1.0 - smoothstep(SQUEEZE_START, VANISH_AT, t)


@dataclass
class SpeedLines:
    """Random set of vertical comic strokes, normalized to the silhouette box."""

    seed: int = 0
    count: int = 70
    x: np.ndarray = field(init=False)
    width: np.ndarray = field(init=False)
    length: np.ndarray = field(init=False)
    speed: np.ndarray = field(init=False)
    phase: np.ndarray = field(init=False)
    order: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.x = rng.random(self.count)
        self.width = rng.choice([1, 2, 2, 3, 3, 4, 5, 7], self.count)
        self.length = rng.uniform(0.18, 0.70, self.count)
        self.speed = rng.uniform(1.6, 4.2, self.count)
        self.phase = rng.random(self.count)
        self.order = rng.random(self.count)  # which lines appear first


def mask_box(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero((mask > 0.25).astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    height, width = mask.shape[:2]
    return (max(0, x - pad), max(0, y - pad),
            min(width, x + w + pad), min(height, y + h + pad))


def _draw_lines(shape: tuple[int, int], lines: SpeedLines, strength: float,
                clock: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (ink, core) coverage layers in 0..1 for the box of ``shape``."""
    height, width = shape
    ink = np.zeros((height, width), np.uint8)
    core = np.zeros((height, width), np.uint8)
    if strength <= 0.0:
        return ink.astype(np.float32), core.astype(np.float32)
    scale = max(1.0, height / 480.0)
    for i in np.flatnonzero(lines.order < strength):
        x = int(lines.x[i] * (width - 1))
        span = lines.length[i]
        # Lines rise through the box and wrap around.
        head = 1.0 - ((lines.phase[i] + clock * lines.speed[i] * 0.5) % (1.0 + span))
        y1 = int(head * height)
        y2 = int((head + span) * height)
        thick = max(1, int(round(lines.width[i] * scale)))
        cv2.line(ink, (x, y1), (x, y2), 255, thick + 2 + int(scale), cv2.LINE_AA)
        # Tapered bright core: shorter than the ink stroke.
        inset = int(span * height * 0.04)
        cv2.line(core, (x, y1 + inset), (x, y2 - inset), 255, thick, cv2.LINE_AA)
    return ink.astype(np.float32) / 255.0, core.astype(np.float32) / 255.0


INK_BGR = np.array((52, 24, 10), np.float32)       # dark navy ink
CORE_BGR = np.array((255, 252, 240), np.float32)   # near-white
GLOW_BGR = np.array((255, 214, 120), np.float32)   # light cyan-blue


def render_teleport(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray | None,
    t: float,
    lines: SpeedLines,
    clock: float,
) -> np.ndarray:
    """Draw the teleport at timeline ``t`` (0 = present, 1 = gone).

    ``mask`` is the person's silhouette (0..1). After the vanish it should be
    the last silhouette seen, so the lines linger where the person stood.
    ``clock`` animates the lines independently of ``t`` so they keep rising
    in the reverse (return) animation too.
    """
    canvas = background.copy()
    if mask is None:
        return canvas if t >= VANISH_AT else frame.copy()
    box = mask_box(mask, pad=28)
    if box is None:
        return canvas if t >= VANISH_AT else frame.copy()
    left, top, right, bottom = box
    region = np.s_[top:bottom, left:right]
    height, width = bottom - top, right - left

    out = canvas[region].astype(np.float32)
    person_mask = mask[region]

    # 1. Person, squeezed horizontally into a thin bright line.
    squeeze = squeeze_amount(t)
    if squeeze > 0.0:
        whiten = smoothstep(0.22, VANISH_AT, t) * 0.85
        person = frame[region].astype(np.float32)
        person = person * (1.0 - whiten) + CORE_BGR * whiten
        alpha = person_mask.copy()
        if squeeze < 1.0:
            new_w = max(1, int(round(width * squeeze)))
            new_h = min(height, int(round(height * (1.0 + 0.10 * (1.0 - squeeze)))))
            person = cv2.resize(person, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Squeeze toward the silhouette's own centre of mass.
            columns = person_mask.sum(axis=0)
            centre = int(np.average(np.arange(width), weights=columns)) if columns.sum() else width // 2
            x0 = int(np.clip(centre - new_w // 2, 0, width - new_w))
            y0 = (height - new_h) // 2
            placed = np.zeros((height, width, 3), np.float32)
            placed_alpha = np.zeros((height, width), np.float32)
            placed[y0:y0 + new_h, x0:x0 + new_w] = person
            placed_alpha[y0:y0 + new_h, x0:x0 + new_w] = alpha
            person, alpha = placed, placed_alpha
            alpha *= min(1.0, squeeze / 0.18)
        a3 = alpha[:, :, None]
        out = person * a3 + out * (1.0 - a3)

    # 2. Comic speed lines confined to (a vertically stretched) silhouette.
    strength = line_strength(t)
    if strength > 0.0:
        ink, core = _draw_lines((height, width), lines, strength, clock)
        stretch = max(3, int(height * 0.10)) | 1
        zone = cv2.dilate(person_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, stretch)))
        zone = cv2.GaussianBlur(zone, (0, 0), 4.0)
        fade = min(1.0, strength * 1.6)
        ink *= zone * fade * 0.85
        core *= zone * fade
        glow = cv2.GaussianBlur(core, (0, 0), 5.0)
        out = out * (1.0 - ink[:, :, None]) + INK_BGR * ink[:, :, None]
        out = out * (1.0 - core[:, :, None]) + CORE_BGR * core[:, :, None]
        out += GLOW_BGR * (glow[:, :, None] * 0.55)

    # 3. The "zip": a bright vertical flash at the instant of vanishing.
    flash = max(0.0, 1.0 - abs(t - VANISH_AT) / 0.07)
    if flash > 0.0:
        columns = person_mask.sum(axis=0)
        centre = int(np.average(np.arange(width), weights=columns)) if columns.sum() else width // 2
        beam = np.zeros((height, width), np.float32)
        cv2.line(beam, (centre, 0), (centre, height - 1), 1.0, max(2, width // 60), cv2.LINE_AA)
        beam = np.clip(beam + cv2.GaussianBlur(beam, (0, 0), 9.0) * 2.5, 0.0, 1.0) * flash
        out = out * (1.0 - beam[:, :, None]) + CORE_BGR * beam[:, :, None]

    canvas[region] = np.clip(out, 0, 255).astype(np.uint8)
    return canvas
