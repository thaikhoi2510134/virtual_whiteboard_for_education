import unittest

import numpy as np

from gestures import GestureStabilizer, PointSmoother
from shapes import create_parametric_shape, scale_shape
from ui import draw_ui


class GestureUtilityTests(unittest.TestCase):
    def test_gesture_requires_stable_duration(self):
        stabilizer = GestureStabilizer(delay_ms=100)
        self.assertEqual(stabilizer.update("DRAW", 0), "NONE")
        self.assertEqual(stabilizer.update("DRAW", 99), "NONE")
        self.assertEqual(stabilizer.update("DRAW", 100), "DRAW")

    def test_point_smoother_reduces_jump(self):
        smoother = PointSmoother(alpha=0.5)
        self.assertEqual(smoother.update("index", (0, 0)), (0, 0))
        self.assertEqual(smoother.update("index", (100, 100)), (50, 50))


class BoardGeometryTests(unittest.TestCase):
    def test_line_scale_updates_measurement(self):
        shape = create_parametric_shape("LINE", (100, 100), (200, 100), (0, 255, 255))
        initial = shape["pts"].copy()
        scale_shape(shape, initial, 2, 1280, 720)
        self.assertEqual(shape["info"], "200px")

    def test_toolbar_fits_small_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tools, colors = draw_ui(frame, "FREE", (0, 255, 255))
        self.assertTrue(all(0 <= region[1][0] < region[1][2] <= 640 for region in tools + colors))


if __name__ == "__main__":
    unittest.main()
