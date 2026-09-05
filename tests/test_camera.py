"""
Unit tests for IP Camera, Frame Buffer, Streamer, and Camera Manager
===================================================================
Tests:
  - URL validation and formatting
  - CameraManager lifecycle and source switching
  - LatestFrameBuffer size=1 drop-oldest behavior
  - Multi-consumer non-blocking read isolation
  - MJPEGStreamer low-latency encoding
  - Diagnostic metrics reporting
"""

import time
import unittest
import numpy as np

from core_ai.video.camera_config import (
    validate_and_format_camera_url,
    IPCameraConfig,
)
from core_ai.video.camera_manager import CameraManager
from core_ai.video.frame_buffer import LatestFrameBuffer
from core_ai.video.ip_camera import IPCamera
from core_ai.video.streamer import MJPEGStreamer


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

    def test_diagnostics_structure(self):
        diag = self.mgr.get_diagnostics()
        self.assertIn("fps", diag)
        self.assertIn("latency_ms", diag)
        self.assertIn("dropped_pct", diag)
        self.assertIn("buffer_size", diag)
        self.assertEqual(diag["buffer_size"], 1)


class TestLatestFrameBuffer(unittest.TestCase):
    def test_single_frame_push_and_read(self):
        buf = LatestFrameBuffer("test-buf")
        f = np.ones((100, 100, 3), dtype=np.uint8) * 42
        t0 = time.time()
        fid = buf.push(f, t0)
        self.assertEqual(fid, 1)

        read_f, read_t, read_id = buf.get_latest()
        self.assertIsNotNone(read_f)
        self.assertEqual(read_f[0, 0, 0], 42)
        self.assertEqual(read_id, 1)
        self.assertEqual(read_t, t0)

        # Non-destructive read: second read still gets the frame
        read_f2, _, _ = buf.get_latest()
        self.assertIsNotNone(read_f2)
        self.assertEqual(read_f2[0, 0, 0], 42)

    def test_drop_oldest_strategy(self):
        buf = LatestFrameBuffer("drop-buf")
        for i in range(10):
            frame = np.ones((50, 50, 3), dtype=np.uint8) * i
            buf.push(frame)

        # Should have dropped 9 frames
        stats = buf.stats
        self.assertEqual(stats["frames_received"], 10)
        self.assertEqual(stats["frames_dropped"], 9)
        self.assertEqual(stats["buffer_size"], 1)

        # Latest frame must be the 9th frame
        cur_f, _, cur_id = buf.get_latest()
        self.assertEqual(cur_f[0, 0, 0], 9)
        self.assertEqual(cur_id, 10)

    def test_multi_consumer_independent_reads(self):
        """Verify that multiple consumers reading at different frequencies do not starve each other."""
        buf = LatestFrameBuffer("multi-consumer")
        f1 = np.ones((20, 20, 3), dtype=np.uint8) * 100
        buf.push(f1)

        # Consumer 1 (Streamer, 30 FPS) reads
        c1_frame, _, _ = buf.get_latest()
        self.assertEqual(c1_frame[0, 0, 0], 100)

        # Consumer 2 (AI, 15 FPS) reads the same latest frame
        c2_frame, _, _ = buf.get_latest()
        self.assertEqual(c2_frame[0, 0, 0], 100)

        # New frame arrives
        f2 = np.ones((20, 20, 3), dtype=np.uint8) * 200
        buf.push(f2)

        # Both consumers see the updated frame independently
        c1_updated, _, _ = buf.get_latest()
        c2_updated, _, _ = buf.get_latest()
        self.assertEqual(c1_updated[0, 0, 0], 200)
        self.assertEqual(c2_updated[0, 0, 0], 200)


class TestIPCameraDropStaleBuffer(unittest.TestCase):
    def test_drop_stale_queue_behavior(self):
        cfg = IPCameraConfig(url="http://localhost:8080/video")
        cam = IPCamera(cfg)

        f1 = np.ones((10, 10, 3), dtype=np.uint8) * 10
        f2 = np.ones((10, 10, 3), dtype=np.uint8) * 20
        f3 = np.ones((10, 10, 3), dtype=np.uint8) * 30

        cam._frame_queue.put_nowait((f1, 100.0))
        cam._frame_queue.put_nowait((f2, 200.0))

        if cam._frame_queue.full():
            cam._frame_queue.get_nowait()
        cam._frame_queue.put_nowait((f3, 300.0))

        read1 = cam.read()
        self.assertEqual(read1[0, 0, 0], 20)
        read2 = cam.read()
        self.assertEqual(read2[0, 0, 0], 30)


class TestMJPEGStreamer(unittest.TestCase):
    def test_fast_encode_and_update(self):
        streamer = MJPEGStreamer(jpeg_quality=70)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        streamer.update(frame)

        jpeg_bytes = streamer.get_latest_jpeg()
        self.assertIsNotNone(jpeg_bytes)
        self.assertTrue(len(jpeg_bytes) > 0)
        # Check standard JPEG header bytes 0xFF, 0xD8
        self.assertEqual(jpeg_bytes[:2], b'\xff\xd8')

        diag = streamer.get_diagnostics()
        self.assertEqual(diag["jpeg_quality"], 70)
        self.assertTrue(diag["encode_latency_ms"] >= 0)


if __name__ == "__main__":
    unittest.main()
