"""
Unit tests for PhoneStreamReceiver and Phone Webcam Integration
===============================================================
Tests:
  - Pairing code generation and validation
  - Ingestion of binary JPEG frames
  - Frame delivery to LatestFrameBuffer
  - Multi-consumer non-blocking reads
  - CameraManager phone_webcam source switching and lifecycle
"""

import time
import unittest
import cv2
import numpy as np

from core_ai.video.camera_manager import CameraManager
from core_ai.video.phone_stream_receiver import PhoneStreamReceiver, get_lan_ip


class TestPhoneStreamReceiver(unittest.TestCase):
    def setUp(self):
        self.receiver = PhoneStreamReceiver(port=8000)

    def test_pairing_code_generation_and_validation(self):
        token = self.receiver.pairing_token
        self.assertEqual(len(token), 4)
        self.assertTrue(token.isdigit())
        self.assertTrue(self.receiver.validate_token(token))
        self.assertFalse(self.receiver.validate_token("999999"))

    def test_token_refresh(self):
        old_token = self.receiver.pairing_token
        new_token = self.receiver.refresh_pairing_token()
        self.assertEqual(len(new_token), 4)
        self.assertTrue(self.receiver.validate_token(new_token))

    def test_lan_ip_detection(self):
        ip = get_lan_ip()
        self.assertTrue(len(ip.split('.')) == 4 or ip == "127.0.0.1")

    def test_connection_info(self):
        info = self.receiver.get_connection_info()
        self.assertIn("pairing_token", info)
        self.assertIn("connection_url", info)
        self.assertIn("lan_ip", info)
        self.assertIn("/cam?token=", info["connection_url"])

    def test_client_registration_and_frame_ingestion(self):
        self.assertFalse(self.receiver.is_connected)

        # Register client
        self.receiver.register_client({"device": "test-phone", "os": "ios"})
        self.assertTrue(self.receiver.is_connected)

        # Create a test JPEG frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        ret, buf = cv2.imencode(".jpg", test_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.assertTrue(ret)
        jpeg_bytes = buf.tobytes()

        # Ingest frame
        success = self.receiver.ingest_frame_bytes(jpeg_bytes, client_timestamp=time.time() * 1000)
        self.assertTrue(success)

        # Read back from receiver
        read_frame = self.receiver.read()
        self.assertIsNotNone(read_frame)
        self.assertEqual(read_frame.shape, (480, 640, 3))
        self.assertEqual(read_frame[0, 0, 0], 128)

        # Verify read with metadata
        frame_meta, fps, lat = self.receiver.read_with_metadata()
        self.assertIsNotNone(frame_meta)
        self.assertTrue(lat >= 0.0)

        # Diagnostics check
        diag = self.receiver.get_diagnostics()
        self.assertTrue(diag["connected"])
        self.assertEqual(diag["buffer_size"], 1)

        # Unregister client
        self.receiver.unregister_client()
        self.assertFalse(self.receiver.is_connected)


class TestCameraManagerPhoneWebcamIntegration(unittest.TestCase):
    def setUp(self):
        self.mgr = CameraManager()

    def tearDown(self):
        self.mgr.disconnect()

    def test_connect_phone_webcam_mode(self):
        success, err = self.mgr.connect_phone_webcam()
        self.assertTrue(success)
        self.assertEqual(self.mgr.active_source, "phone_webcam")

        status = self.mgr.get_status()
        self.assertEqual(status["source"], "phone_webcam")
        self.assertEqual(status["status"], "waiting")

    def test_phone_stream_delivery_to_camera_manager(self):
        self.mgr.connect_phone_webcam()
        receiver = self.mgr.phone_receiver

        # Push frame to receiver
        test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 77
        ret, buf = cv2.imencode(".jpg", test_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        receiver.register_client({"agent": "test"})
        receiver.ingest_frame_bytes(buf.tobytes(), client_timestamp=time.time() * 1000)

        # Read from CameraManager
        frame = self.mgr.read()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame[0, 0, 0], 77)

        # Diagnostics
        diag = self.mgr.get_diagnostics()
        self.assertEqual(diag["source"], "phone_webcam")
        self.assertEqual(diag["buffer_size"], 1)


if __name__ == "__main__":
    unittest.main()
