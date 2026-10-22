"""Unit tests for Step 5: Score fusion (1-100) and label boundaries."""

from app.services.fusion import compute_score_and_label


def test_no_face_found_returns_null_score():
    score, label, reasons = compute_score_and_label(
        metadata_score=50,
        face_max_score=None,
        full_image_score=30,
        faces_found=0,
        metadata_detail="No camera EXIF",
    )
    assert score is None
    assert label == "NO_FACE_FOUND"
    assert any("No human face detected" in r for r in reasons)


def test_likely_real_label_boundary():
    # Low metadata score (15), low face score (10), low full image (20)
    score, label, reasons = compute_score_and_label(
        metadata_score=15,
        face_max_score=10,
        full_image_score=20,
        faces_found=1,
        metadata_detail="Camera EXIF verified",
        has_camera_exif=True,
    )
    assert score is not None
    assert 1 <= score <= 30
    assert label == "LIKELY_REAL"
    assert any("natural human features" in r for r in reasons)


def test_uncertain_label_boundary():
    # Moderate scores (50, 50, 50)
    score, label, reasons = compute_score_and_label(
        metadata_score=50,
        face_max_score=50,
        full_image_score=50,
        faces_found=1,
        metadata_detail="No camera EXIF",
    )
    assert score is not None
    assert 31 <= score <= 70
    assert label == "UNCERTAIN"


def test_likely_fake_label_boundary():
    # High metadata score (85), high face score (90), high full image (80)
    score, label, reasons = compute_score_and_label(
        metadata_score=85,
        face_max_score=90,
        full_image_score=80,
        faces_found=1,
        metadata_detail="AI tool signature in metadata",
        tool_detected="Stable Diffusion Parameters",
    )
    assert score is not None
    assert 71 <= score <= 100
    assert label == "LIKELY_FAKE"
    assert any("strongly suggests AI generation" in r for r in reasons)
    assert any("AI tool signature" in r for r in reasons)


def test_score_clamping():
    # Theoretical extreme 0 or >100 clamped to [1, 100]
    score_low, _, _ = compute_score_and_label(
        metadata_score=0,
        face_max_score=0,
        full_image_score=0,
        faces_found=1,
        metadata_detail="test",
    )
    assert score_low == 1

    score_high, _, _ = compute_score_and_label(
        metadata_score=150,
        face_max_score=150,
        full_image_score=150,
        faces_found=1,
        metadata_detail="test",
    )
    assert score_high == 100
