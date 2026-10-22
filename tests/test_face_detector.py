"""Unit tests for Step 3: Face detection with margin cropping."""

from PIL import Image
from app.services.face_detector import YuNetDetector


def test_detect_no_face_in_blank_image():
    detector = YuNetDetector(model_path="models/nonexistent_model.onnx")
    blank = Image.new("RGB", (50, 50), color=(0, 0, 0))
    crops = detector.detect_and_crop(blank)
    assert len(crops) == 0


def test_fallback_portrait_detection():
    detector = YuNetDetector(model_path="models/nonexistent_model.onnx")
    # Using mock face test marker
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img._test_mock_face = True

    crops = detector.detect_and_crop(img)
    assert len(crops) == 1
    fc = crops[0]
    assert len(fc.box) == 4
    # Box should be [x, y, w, h]
    bx, by, bw, bh = fc.box
    assert bw > 0 and bh > 0
    # Crop image should be larger than original box due to 25% margin ratio
    cw, ch = fc.crop.size
    assert cw >= bw
    assert ch >= bh


def test_margin_cropping_bounds():
    detector = YuNetDetector(model_path="models/nonexistent_model.onnx")
    detector.crop_margin_ratio = 0.50
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    img._test_mock_face = True

    crops = detector.detect_and_crop(img)
    assert len(crops) == 1
    # Check that crop does not exceed image bounds
    cw, ch = crops[0].crop.size
    assert cw <= 100
    assert ch <= 100
