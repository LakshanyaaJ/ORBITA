"""
Unit tests for IP Camera and Camera Manager
===========================================
Tests:
  - URL validation and formatting (discrete IP/port/path and direct stream URLs)
  - CameraManager lifecycle and source switching
  - Error handling for unreachable IP camera streams
"""

import sys
import unittest
import numpy as np

import core_ai

from core_ai.video.camera_config import (
    validate_and_format_camera_url,
    IPCameraConfig,
)
from core_ai.video.camera_manager import CameraManager
from core_ai.video.ip_camera import IPCamera


class TestCameraConfigValidation(unittest.TestCase):
    def test_complete_valid_url(self):
        valid, url, err = validate_and_format_camera_url(url="http://192.168.1.105:8080/video")
        self.assertTrue(valid)
        self.assertEqual(url, "http://192.168.1.105:8080/video")
        self.assertEqual(err, "")

    def test_url_without_scheme_adds_http(self):
        valid, url, err = validate_and_format_camera_url(url="192.168.1.105:8080/video")
        self.assertTrue(valid)
        self.assertEqual(url, "http://192.168.1.105:8080/video")

    def test_rtsp_url(self):
        valid, url, err = validate_and_format_camera_url(url="rtsp://192.168.1.105:554/h264")
        self.assertTrue(valid)
        self.assertEqual(url, "rtsp://192.168.1.105:554/h264")

    def test_invalid_scheme(self):
        valid, url, err = validate_and_format_camera_url(url="ftp://192.168.1.105:8080/video")
        self.assertFalse(valid)
        self.assertIn("Unsupported stream protocol", err)

    def test_discrete_fields_building(self):
        valid, url, err = validate_and_format_camera_url(
            ip="192.168.1.50", port=8080, path="/video"
        )
        self.assertTrue(valid)
        self.assertEqual(url, "http://192.168.1.50:8080/video")

    def test_discrete_fields_cleans_input(self):
        valid, url, err = validate_and_format_camera_url(
            ip="http://192.168.1.50/", port="8081", path="stream.mjpg"
        )
        self.assertTrue(valid)
        self.assertEqual(url, "http://192.168.1.50:8081/stream.mjpg")

    def test_invalid_ip_format(self):
        valid, url, err = validate_and_format_camera_url(ip="invalid..ip??", port=8080)
        self.assertFalse(valid)
        self.assertIn("Invalid IP address", err)

    def test_invalid_port_range(self):
        valid, url, err = validate_and_format_camera_url(ip="192.168.1.1", port=999999)
        self.assertFalse(valid)
        self.assertIn("Port must be between", err)


class TestCameraManagerLifecycle(unittest.TestCase):
    def setUp(self):
        self.mgr = CameraManager()

    def tearDown(self):
        self.mgr.disconnect()

    def test_initial_simulation_mode(self):
        status = self.mgr.get_status()
        self.assertTrue(status["connected"])
        self.assertEqual(status["source"], "sim")

    def test_disconnect(self):
        self.mgr.disconnect()
        status = self.mgr.get_status()
        self.assertFalse(status["connected"])
        self.assertEqual(status["source"], "disconnected")

    def test_invalid_ip_connection_fails_cleanly(self):
        success, err = self.mgr.connect_ip_camera(
            url="http://127.0.0.1:59999/nonexistent_stream",
            timeout_sec=0.5,
        )
        self.assertFalse(success)
        status = self.mgr.get_status()
        self.assertFalse(status["connected"])
        self.assertEqual(status["status"], "error")


class TestIPCameraDropStaleBuffer(unittest.TestCase):
    def test_drop_stale_queue_behavior(self):
        cfg = IPCameraConfig(url="http://localhost:8080/video")
        cam = IPCamera(cfg)
        
        # Test internal queue drops oldest frame when full
        f1 = np.ones((10, 10, 3), dtype=np.uint8) * 10
        f2 = np.ones((10, 10, 3), dtype=np.uint8) * 20
        f3 = np.ones((10, 10, 3), dtype=np.uint8) * 30

        cam._frame_queue.put_nowait((f1, 100.0))
        cam._frame_queue.put_nowait((f2, 200.0))
        
        # When queue is full (maxsize=2), simulate worker dropping oldest
        if cam._frame_queue.full():
            cam._frame_queue.get_nowait()
        cam._frame_queue.put_nowait((f3, 300.0))
        
        # Should now read f2 and then f3 (f1 dropped)
        read1 = cam.read()
        self.assertEqual(read1[0, 0, 0], 20)
        read2 = cam.read()
        self.assertEqual(read2[0, 0, 0], 30)


if __name__ == "__main__":
    unittest.main()
