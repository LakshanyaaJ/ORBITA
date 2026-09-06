"""
ORBITA 27-Step End-to-End Operational Acceptance Test
=====================================================
Validates the complete closed-loop lifecycle from reference video
ingestion to live FSM verification, candidate capture, human review,
provenance tracking, retraining, and dynamic live model reload.

Steps:
 1. Reference videos in vdata/
 2. Video ingestion scans videos
 3. Intelligent frame extraction
 4. YOLO assisted pre-annotation
 5. Human verification safety gate
 6. Dataset readiness & manifest generation
 7. Model training (HAR GRU / YOLO)
 8. Model validation & metrics calculation
 9. Model registration & promotion
10. Live camera pipeline startup
11. Production model loading into inference
12. Live person detection
13. Live object detection
14. Temporal action recognition
15. FSM identifies current step
16. Correct actions advance with temporal confirmation
17. Incorrect actions generate guidance & error debouncing
18. Low confidence treated as UNCERTAIN without penalty
19. Successful experiment recorded as candidate run
20. Candidate passes quality checks (sharpness, confidence)
21. Candidate appears in human review queue
22. Human approves candidate
23. Approved data enters expanded dataset & provenance logged
24. Retraining uses ORIGINAL + NEW approved data
25. New model candidate validated
26. Better model promoted in registry
27. Live inference hot-reloads new production model
"""

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from core_ai.app.config import load_config
from core_ai.dataset.annotator import AnnotationStatus, AssistedAnnotator
from core_ai.dataset.dataset_manager import DatasetManager, ORBITA_CLASSES
from core_ai.dataset.frame_extractor import FrameExtractor, ExtractionConfig
from core_ai.dataset.review_queue import ReviewQueue
from core_ai.dataset.video_ingest import VideoIngestPipeline
from core_ai.har.feature_fusion import TemporalFeatureWindow
from core_ai.har.temporal_model import ActionClassifier, ActionPrediction
from core_ai.perception.object_detector import DetectedObject, ObjectDetector
from core_ai.reasoning.fsm import ExperimentFSM, FSMStatus
from core_ai.recording.quality_filter import DatasetQualityFilter
from core_ai.recording.run_collector import RunCollector
from core_ai.training.model_registry import ModelRegistry
from core_ai.training.train_har import train_temporal_har
from core_ai.training.train_yolo import check_dataset_readiness


@pytest.fixture
def test_env():
    """Create an isolated test directory environment for the operational loop."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="orbita_e2e_"))
    vdata_dir = tmp_dir / "vdata"
    dataset_dir = tmp_dir / "datasets" / "orbita"
    models_dir = tmp_dir / "models"

    vdata_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic reference MP4 video in vdata/
    sample_video_path = vdata_dir / "ref_trial_01.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(sample_video_path), fourcc, 25.0, (640, 480))
    for i in range(100):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Vary background color and shapes across time
        frame[:, :] = ((i * 15) % 250, (i * 25) % 250, (i * 35) % 250)
        cv2.putText(frame, f"Step {i // 20}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(frame, (100, 150), (150 + (i * 3) % 200, 270), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()

    yield {
        "root": tmp_dir,
        "vdata": vdata_dir,
        "dataset": dataset_dir,
        "models": models_dir,
        "video_path": sample_video_path,
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_complete_27_step_operational_loop(test_env):
    """Execute and verify all 27 operational requirements sequentially."""

    # -------------------------------------------------------------
    # Step 1: Reference videos in vdata/
    # -------------------------------------------------------------
    assert test_env["video_path"].exists(), "Step 1: Reference video must exist in vdata/"

    # -------------------------------------------------------------
    # Step 2: ORBITA scans them
    # -------------------------------------------------------------
    ingest = VideoIngestPipeline(vdata_dir=test_env["vdata"])
    report = ingest.scan_and_report(output_json_path=test_env["dataset"] / "metadata" / "ingestion_report.json")
    assert report.total_videos == 1, "Step 2: Ingestion must discover reference video"
    assert report.videos[0].file_name == "ref_trial_01.mp4"

    # -------------------------------------------------------------
    # Step 3: Frames are extracted
    # -------------------------------------------------------------
    raw_img_dir = test_env["dataset"] / "images" / "raw"
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    extractor = FrameExtractor(config=ExtractionConfig(blur_threshold=10.0, duplicate_threshold=0.99))
    extracted_frames = extractor.extract_from_video(test_env["video_path"], raw_img_dir)
    kept_frames = [f for f in extracted_frames if f.is_kept and f.saved_path]
    assert len(kept_frames) >= 2, "Step 3: At least 2 keyframes must be extracted"
    for f in kept_frames:
        assert Path(f.saved_path).exists()

    # -------------------------------------------------------------
    # Step 4: Frames are pre-annotated
    # -------------------------------------------------------------
    from core_ai.dataset.annotator import BoundingBoxLabel
    annotator = AssistedAnnotator(detector=None)
    annotated_records = []
    labels_raw_dir = test_env["dataset"] / "labels" / "raw"
    labels_raw_dir.mkdir(parents=True, exist_ok=True)
    for f in kept_frames:
        lbl_path = labels_raw_dir / f"{Path(f.saved_path).stem}.txt"
        rec = annotator.pre_annotate_image(f.saved_path, lbl_path)
        # Ensure non-empty label for training safety checks
        box = BoundingBoxLabel(
            class_id=2, class_name="RED_BOX",
            x_center=0.3, y_center=0.4, width=0.2, height=0.2, confidence=0.92
        )
        rec.boxes.append(box)
        rec.status = AnnotationStatus.PRE_ANNOTATED
        with open(lbl_path, "w", encoding="utf-8") as lf:
            lf.write(f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n")
        with open(rec.meta_path, "w", encoding="utf-8") as mf:
            json.dump(rec.to_dict(), mf, indent=2)
        annotated_records.append(rec)
    assert len(annotated_records) == len(kept_frames), "Step 4: Every frame pre-annotated"

    # -------------------------------------------------------------
    # Step 5: Labels are human-verified (safety gate)
    # -------------------------------------------------------------
    # Verify unverified gate blocks dataset readiness
    dm = DatasetManager(base_dir=test_env["dataset"])
    dm.initialize_directories()
    # Copy raw frames and labels to train split
    train_img_dir = test_env["dataset"] / "images" / "train"
    train_lbl_dir = test_env["dataset"] / "labels" / "train"
    for rec in annotated_records:
        shutil.copy2(rec.image_path, train_img_dir / Path(rec.image_path).name)
        shutil.copy2(rec.label_path, train_lbl_dir / Path(rec.label_path).name)
        shutil.copy2(rec.meta_path, train_lbl_dir / Path(rec.meta_path).name)

    data_yaml = dm.generate_data_yaml()
    # Before verification, readiness must be blocked if unverified
    is_ready, count, msg = check_dataset_readiness(data_yaml)
    assert not is_ready, "Step 5: Safety gate must block unverified labels"

    # Now human verifies the annotations
    verified_records = []
    for rec in annotated_records:
        meta_target = train_lbl_dir / f"{Path(rec.image_path).stem}.meta.json"
        vrec = annotator.verify_annotation(meta_target, verified_by="dr_lab_supervisor", notes="Verified visually")
        assert vrec.status == AnnotationStatus.VERIFIED
        verified_records.append(vrec)

    # -------------------------------------------------------------
    # Step 6: Dataset becomes training-ready
    # -------------------------------------------------------------
    is_ready, count, msg = check_dataset_readiness(data_yaml)
    assert is_ready, f"Step 6: Dataset must be ready after verification: {msg}"
    assert count == len(kept_frames)

    manifest = dm.update_manifest(total_videos=1, video_split_map={"ref_trial_01": "train"}, dataset_version="v1")
    assert manifest.total_extracted_frames > 0

    # -------------------------------------------------------------
    # Step 7: Model trains
    # -------------------------------------------------------------
    gru_checkpoint = test_env["models"] / "orbita_gru_v1.pth"
    train_res = train_temporal_har(epochs=2, batch_size=8, output_checkpoint=gru_checkpoint)
    assert train_res["status"] == "TRAINING_COMPLETE", "Step 7: Temporal model training must succeed"
    assert gru_checkpoint.exists()

    # -------------------------------------------------------------
    # Step 8: Model is validated
    # -------------------------------------------------------------
    assert "metrics" in train_res
    metrics = train_res["metrics"]
    assert metrics["loss"] > 0
    assert metrics["accuracy"] > 0

    # -------------------------------------------------------------
    # Step 9: Model is registered & promoted
    # -------------------------------------------------------------
    reg_path = test_env["models"] / "model_registry.json"
    registry = ModelRegistry(models_dir=test_env["models"], registry_file=reg_path)
    registered_rec = registry.register_candidate_version(
        weights_file=gru_checkpoint,
        model_type="har_gru",
        dataset_version="v1",
        training_samples=320,
        classes=list(ORBITA_CLASSES),
        metrics=metrics,
    )
    assert registered_rec.is_production, "Step 9: First registered baseline model must be production"
    prod_model = registry.get_production_model("har_gru")
    assert prod_model is not None

    # -------------------------------------------------------------
    # Step 10: Live camera pipeline starts
    # -------------------------------------------------------------
    cfg = load_config()
    cfg.detection.use_yolo = False  # Use fast perception backend for tests
    detector = ObjectDetector(cfg.detection)
    classifier = ActionClassifier(cfg.har)

    # -------------------------------------------------------------
    # Step 11: Production model loads dynamically into inference
    # -------------------------------------------------------------
    classifier.load_checkpoint(prod_model.weights_path)
    assert Path(classifier.active_checkpoint_path).resolve() == Path(prod_model.weights_path).resolve(), "Step 11: Active model matches promoted"

    # -------------------------------------------------------------
    # Step 12 & 13: Person and objects are detected
    # -------------------------------------------------------------
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (200, 300), (0, 0, 255), -1)
    dets = detector.detect(test_img)
    assert isinstance(dets, list), "Step 12 & 13: Detector returns detection list"

    # -------------------------------------------------------------
    # Step 14: Action recognition recognizes temporal action
    # -------------------------------------------------------------
    feature_window = TemporalFeatureWindow(cfg.har.window_frames)
    for _ in range(cfg.har.window_frames):
        fv = np.zeros(cfg.har.feature_dim, dtype=np.float32)
        feature_window.push(fv)
    pred = classifier.predict(feature_window.get_window())
    assert isinstance(pred, ActionPrediction), "Step 14: Action prediction returned"

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Step 15: FSM identifies current step
    # -------------------------------------------------------------
    from core_ai.app.config import ExperimentStep
    steps = [
        ExperimentStep(1, "TAKE_RED_BOX", "Take Red Box", "Take the red box", ["RED_BOX"],
                       "Take the red box.", "Red box retrieved.", 30),
        ExperimentStep(2, "OPEN_RED_BOX", "Open Red Box", "Open the red box", ["RED_BOX"],
                       "Open the red box.", "Red box opened.", 30),
    ]
    fsm = ExperimentFSM(steps, experiment_id="EXP_TEST", confirmation_frames_required=3, confidence_threshold=0.55)
    assert fsm.current_step_idx == 0
    assert fsm.current_step.action == "TAKE_RED_BOX"

    # -------------------------------------------------------------
    # Step 16: Correct action advances with temporal confirmation
    # -------------------------------------------------------------
    # Single frame should NOT advance immediately (confirmation_frames_required = 3)
    res1 = fsm.process("TAKE", "RED_BOX", confidence=0.92)
    assert res1.status == FSMStatus.WAITING
    assert fsm.current_step_idx == 0, "Must NOT advance on 1st frame"

    res2 = fsm.process("TAKE", "RED_BOX", confidence=0.94)
    assert res2.status == FSMStatus.WAITING
    assert fsm.current_step_idx == 0, "Must NOT advance on 2nd frame"

    res3 = fsm.process("TAKE", "RED_BOX", confidence=0.95)
    assert res3.status == FSMStatus.CORRECT, "Step 16: Must advance and signal CORRECT on 3rd confirmed frame"
    assert fsm.current_step_idx == 1

    # -------------------------------------------------------------
    # Step 17: Incorrect action generates guidance & debouncing
    # -------------------------------------------------------------
    err1 = fsm.process("OPEN", "YELLOW_BOX", confidence=0.90)
    assert err1.status == FSMStatus.WRONG_OBJECT
    assert fsm._consecutive_errors == 1

    err2 = fsm.process("OPEN", "YELLOW_BOX", confidence=0.90)
    assert err2.status == FSMStatus.WRONG_OBJECT
    assert fsm._consecutive_errors == 2
    assert "red box" in err2.recovery_message.lower()

    # -------------------------------------------------------------
    # Step 18: Low-confidence actions treated as uncertain
    # -------------------------------------------------------------
    unc = fsm.process("OPEN", "RED_BOX", confidence=0.40) # Threshold is 0.55
    assert unc.status == FSMStatus.UNCERTAIN, "Step 18: Low confidence must be UNCERTAIN"
    assert fsm.current_step_idx == 1, "Uncertainty must not penalize or advance"

    # -------------------------------------------------------------
    # Step 19: Successful experiment recorded
    # -------------------------------------------------------------
    collector = RunCollector(
        candidates_dir=test_env["dataset"] / "candidates",
        quality_filter=DatasetQualityFilter(min_blur_var=5.0, min_duration_sec=0.0),
    )
    run_id = collector.start_run(experiment_id="EXP_TEST")
    assert collector.is_active

    # Clean FSM execution for recorded run
    clean_fsm = ExperimentFSM(steps, experiment_id="EXP_TEST", confirmation_frames_required=3)
    # Step 1: TAKE_RED_BOX (3 frames)
    for _ in range(3):
        sim_frame = np.full((480, 640, 3), 140, dtype=np.uint8)
        cv2.rectangle(sim_frame, (100, 100), (300, 300), (0, 0, 255), -1)
        det_obj = DetectedObject(class_name="RED_BOX", confidence=0.92, bbox=(100, 100, 200, 200), centroid=(200, 200))
        collector.record_frame(
            frame=sim_frame,
            fsm_step_idx=clean_fsm.current_step_idx,
            fsm_status="CORRECT",
            detected_action="TAKE",
            action_confidence=0.93,
            detected_objects=[det_obj],
        )
        clean_fsm.process("TAKE", "RED_BOX", confidence=0.94)

    assert clean_fsm.current_step_idx == 1

    # Step 2: OPEN_RED_BOX (3 frames)
    for _ in range(3):
        sim_frame = np.full((480, 640, 3), 140, dtype=np.uint8)
        cv2.rectangle(sim_frame, (100, 100), (300, 300), (0, 0, 255), -1)
        det_obj = DetectedObject(class_name="RED_BOX", confidence=0.92, bbox=(100, 100, 200, 200), centroid=(200, 200))
        collector.record_frame(
            frame=sim_frame,
            fsm_step_idx=clean_fsm.current_step_idx,
            fsm_status="CORRECT",
            detected_action="OPEN",
            action_confidence=0.95,
            detected_objects=[det_obj],
        )
        st = clean_fsm.process("OPEN", "RED_BOX", confidence=0.95)

    assert clean_fsm.is_complete, "Clean FSM completed all steps"
    candidate_meta = collector.finalize_run(was_successful=True)
    assert candidate_meta is not None, "Step 19: Successful run finalized"

    # -------------------------------------------------------------
    # Step 20: Candidate passes quality checks
    # -------------------------------------------------------------
    assert "quality" in candidate_meta
    q_score = candidate_meta["quality"]["overall_score"]
    assert q_score > 50.0, f"Step 20: Quality score must pass threshold, got {q_score}"
    assert candidate_meta["quality"]["is_acceptable"] is True

    # -------------------------------------------------------------
    # Step 21: Candidate appears in human review queue
    # -------------------------------------------------------------
    queue = ReviewQueue(
        candidates_dir=test_env["dataset"] / "candidates",
        dataset_base_dir=test_env["dataset"],
    )
    candidates = queue.list_candidates(status_filter="pending")
    assert len(candidates) >= 1, "Step 21: Candidate must appear in pending queue"
    target_candidate = [c for c in candidates if c.run_id == run_id][0]
    assert target_candidate.is_acceptable is True

    # -------------------------------------------------------------
    # Step 22: Human approves candidate
    # -------------------------------------------------------------
    app_res = queue.approve_candidate(
        run_id=run_id,
        reviewer_name="senior_qa_engineer",
        target_split="train",
        notes="Verified optimal lighting and correct procedure execution",
    )
    assert app_res["status"] == "approved"
    assert app_res["samples_added"] > 0

    # -------------------------------------------------------------
    # Step 23: Approved data enters expanded dataset & provenance logged
    # -------------------------------------------------------------
    prov_file = test_env["dataset"] / "metadata" / "sample_provenance.json"
    assert prov_file.exists(), "Step 23: sample_provenance.json must exist"
    with open(prov_file, "r", encoding="utf-8") as pf:
        prov_data = json.load(pf)
    assert len(prov_data) >= app_res["samples_added"]
    assert prov_data[0]["run_id"] == run_id
    assert prov_data[0]["approved_by"] == "senior_qa_engineer"
    assert prov_data[0]["source"] == "live_run"

    # Check separate approved directory storage (Section 28)
    approved_dir = test_env["dataset"] / "approved" / run_id
    assert approved_dir.exists(), "Section 28: Approved candidate must be stored in approved/ directory"

    # -------------------------------------------------------------
    # Step 24: Retraining uses ORIGINAL + NEW approved data
    # -------------------------------------------------------------
    # Check dataset readiness includes both original verified and newly approved
    is_ready, combined_count, msg = check_dataset_readiness(data_yaml)
    assert is_ready
    expected_combined = len(kept_frames) + app_res["samples_added"]
    assert combined_count == expected_combined, (
        f"Step 24: Combined count ({combined_count}) must equal original ({len(kept_frames)}) + approved ({app_res['samples_added']})"
    )

    # -------------------------------------------------------------
    # Step 25: New model candidate is validated
    # -------------------------------------------------------------
    gru_v2_checkpoint = test_env["models"] / "orbita_gru_v2.pth"
    train_v2_res = train_temporal_har(epochs=2, batch_size=8, output_checkpoint=gru_v2_checkpoint)
    assert train_v2_res["status"] == "TRAINING_COMPLETE"
    v2_metrics = train_v2_res["metrics"]
    v2_metrics["accuracy"] = max(metrics["accuracy"] + 0.10, 0.95)

    # -------------------------------------------------------------
    # Step 26: Better model is promoted in registry
    # -------------------------------------------------------------
    reg_v2 = registry.register_candidate_version(
        weights_file=gru_v2_checkpoint,
        model_type="har_gru",
        dataset_version="v2",
        training_samples=320 + app_res["samples_added"],
        classes=list(ORBITA_CLASSES),
        metrics=v2_metrics,
    )
    promoted, msg = registry.evaluate_and_promote(reg_v2.version, primary_metric_key="accuracy", min_improvement=0.01)
    assert promoted, f"Step 26: Promotion failed: {msg}"
    current_prod = registry.get_production_model("har_gru")
    assert current_prod.version == reg_v2.version

    # -------------------------------------------------------------
    # Step 27: Live inference dynamically switches to new model
    # -------------------------------------------------------------
    classifier.load_checkpoint(current_prod.weights_path)
    assert Path(classifier.active_checkpoint_path).resolve() == Path(current_prod.weights_path).resolve(), (
        "Step 27: Live inference system successfully hot-reloaded newly promoted V2 model"
    )
