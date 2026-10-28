import unittest

import numpy as np

from goku_effect import (
    VANISH_AT, FaceBox, GestureHold, SpeedLines, fingers_on_forehead, goku_pose,
    render_teleport, two_fingers_together,
)


def hand(dx: float = 0.0, dy: float = 0.0, spread: float = 0.0, ring_open: bool = False) -> np.ndarray:
    """21 hand landmarks in pixels, fingers pointing up (y grows downward)."""
    points = np.zeros((21, 2), np.float32)
    points[0] = (300, 400)                                      # wrist
    points[1:5] = [(270, 385), (250, 360), (240, 340), (235, 325)]  # thumb
    points[5:9] = [(285, 310), (285 - spread * .4, 270), (285 - spread * .7, 245), (285 - spread, 220)]
    points[9:13] = [(305, 305), (305 + spread * .4, 262), (305 + spread * .7, 235), (305 + spread, 208)]
    if ring_open:
        points[13:17] = [(322, 310), (324, 272), (325, 248), (326, 225)]
    else:
        points[13:17] = [(322, 310), (326, 285), (322, 305), (318, 318)]
    points[17:21] = [(336, 318), (340, 298), (336, 312), (332, 322)]
    return points + np.array([dx, dy], np.float32)


FACE = FaceBox(x1=230, y1=200, x2=380, y2=370, eye_y=260)


class GestureTests(unittest.TestCase):
    def test_two_fingers_together_on_forehead(self):
        self.assertTrue(two_fingers_together(hand()))
        self.assertTrue(fingers_on_forehead(hand(), FACE))
        self.assertTrue(goku_pose([hand()], FACE))

    def test_v_sign_is_rejected(self):
        self.assertFalse(two_fingers_together(hand(spread=40)))

    def test_ring_finger_extended_is_rejected(self):
        self.assertFalse(two_fingers_together(hand(ring_open=True)))

    def test_hand_away_from_forehead_is_rejected(self):
        self.assertFalse(goku_pose([hand(dy=180)], FACE))   # in front of the chest
        self.assertFalse(goku_pose([hand(dx=300)], FACE))   # beside the head
        self.assertFalse(goku_pose([hand()], None))         # no face


class HoldTests(unittest.TestCase):
    def test_fires_after_hold_and_needs_release_to_rearm(self):
        hold = GestureHold(hold_seconds=1.0, release_seconds=0.25)
        fired = [hold.update(True, t / 10) for t in range(0, 11)]
        self.assertEqual(fired.count(True), 1)
        self.assertTrue(fired[10])
        # Keeping the hand up does not fire again.
        self.assertFalse(any(hold.update(True, 1.0 + t / 10) for t in range(1, 30)))
        # Lower the hand, then hold again for a second.
        for t in range(0, 4):
            hold.update(False, 4.0 + t / 10)
        self.assertTrue(hold.armed)
        fired = [hold.update(True, 5.0 + t / 10) for t in range(0, 11)]
        self.assertTrue(fired[10])

    def test_breaking_the_pose_restarts_countdown(self):
        hold = GestureHold(hold_seconds=1.0, grace_seconds=0.15)
        for t in range(0, 8):
            self.assertFalse(hold.update(True, t / 10))
        hold.update(False, 0.8)
        hold.update(False, 1.0)   # dropout longer than the grace period
        self.assertFalse(hold.update(True, 1.1))
        self.assertFalse(hold.update(True, 1.9))
        self.assertTrue(hold.update(True, 2.1))

    def test_brief_dropout_is_tolerated(self):
        hold = GestureHold(hold_seconds=1.0, grace_seconds=0.15)
        for t in range(0, 6):
            hold.update(True, t / 10)
        hold.update(False, 0.55)
        self.assertTrue(hold.update(True, 1.0))


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((240, 320, 3), 40, np.uint8)
        self.frame = np.full((240, 320, 3), 200, np.uint8)
        self.mask = np.zeros((240, 320), np.float32)
        self.mask[60:220, 130:190] = 1.0
        self.lines = SpeedLines(seed=3)

    def test_start_shows_the_person(self):
        out = render_teleport(self.frame, self.background, self.mask, 0.0, self.lines, 0.0)
        self.assertEqual(out.shape, self.frame.shape)
        self.assertEqual(int(out[140, 160, 0]), 200)
        self.assertEqual(int(out[10, 10, 0]), 40)

    def test_lines_appear_over_the_silhouette(self):
        out = render_teleport(self.frame, self.background, self.mask, 0.35, self.lines, 0.2)
        region = out[60:220, 130:190].astype(int)
        self.assertGreater(np.abs(region - 200).max(), 60)

    def test_person_gone_after_vanish_and_lines_fade(self):
        late = render_teleport(self.frame, self.background, self.mask, VANISH_AT + 0.12,
                               self.lines, 0.7)
        # No pure-red person pixels remain: only background, ink and white lines.
        red = np.zeros_like(self.frame)
        red[:, :, 2] = 220
        late = render_teleport(red, self.background, self.mask, VANISH_AT + 0.12, self.lines, 0.7)
        b, g, r = (late[:, :, i].astype(int) for i in range(3))
        self.assertFalse(np.any((r > 150) & (g < 90) & (b < 90)))
        done = render_teleport(self.frame, self.background, self.mask, 1.0, self.lines, 1.1)
        np.testing.assert_array_equal(done, self.background)

    def test_empty_mask_does_not_crash(self):
        empty = np.zeros_like(self.mask)
        out = render_teleport(self.frame, self.background, empty, 0.3, self.lines, 0.1)
        np.testing.assert_array_equal(out, self.frame)


if __name__ == "__main__":
    unittest.main()
