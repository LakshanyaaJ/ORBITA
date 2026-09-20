"""
Automated Ontology & Feature Contract Consistency Test
======================================================
Validates that object detector classes, feature fusion ontology slots,
and FSM expected objects are semantically aligned and free of index collisions.
"""

import unittest
import numpy as np

from core_ai.perception.object_detector import REQUIRED_YOLO_CLASSES, map_raw_to_app_class
from core_ai.har.feature_fusion import OBJECT_CLASSES, get_object_class_idx, FEATURE_DIM, build_feature_vector


class TestOntologyAndFeatureContract(unittest.TestCase):

    def test_detector_ontology_has_7_canonical_classes(self):
        self.assertEqual(len(REQUIRED_YOLO_CLASSES), 7)
        expected = ["LOCATION_A", "LOCATION_B", "PEN", "WATCH", "BLUE_BOX", "YELLOW_BOX", "HAND"]
        for cls_name in expected:
            self.assertIn(cls_name, REQUIRED_YOLO_CLASSES)

    def test_feature_fusion_object_slots_are_unique_and_collision_free(self):
        mapped_slots = {}
        for cls_name in REQUIRED_YOLO_CLASSES:
            slot = get_object_class_idx(cls_name)
            self.assertLess(slot, 7, f"Class {cls_name} mapped to out-of-range slot {slot}")
            self.assertNotIn(slot, mapped_slots.values(), f"Slot collision: {cls_name} maps to slot {slot} already taken by {mapped_slots}")
            mapped_slots[cls_name] = slot

        # Verify exact slot assignments
        self.assertEqual(get_object_class_idx("LOCATION_A"), 0)
        self.assertEqual(get_object_class_idx("LOCATION_B"), 1)
        self.assertEqual(get_object_class_idx("PEN"), 2)
        self.assertEqual(get_object_class_idx("WATCH"), 3)
        self.assertEqual(get_object_class_idx("BLUE_BOX"), 4)
        self.assertEqual(get_object_class_idx("YELLOW_BOX"), 5)
        self.assertEqual(get_object_class_idx("HAND"), 6)

    def test_feature_vector_contract(self):
        fv = build_feature_vector(None, None, None, [], [], (720, 1280))
        self.assertEqual(fv.shape, (64,))
        self.assertTrue(np.isfinite(fv).all(), "Feature vector contains NaN or Inf values")


if __name__ == "__main__":
    unittest.main()
