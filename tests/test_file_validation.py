import pytest
from src.utils.file_validation import validate_image_file, validate_video_file

def test_valid_image():
    valid, msg = validate_image_file("worker.jpg", 1024 * 1024)
    assert valid is True
    assert "Valid" in msg

def test_invalid_image_extension():
    valid, msg = validate_image_file("document.pdf", 500)
    assert valid is False
    assert "Unsupported" in msg

def test_oversized_image():
    valid, msg = validate_image_file("large.png", 30 * 1024 * 1024, max_mb=25)
    assert valid is False
    assert "exceeds" in msg

def test_valid_video():
    valid, msg = validate_video_file("cctv_feed.mp4", 50 * 1024 * 1024)
    assert valid is True

def test_invalid_video_extension():
    valid, msg = validate_video_file("audio.mp3", 1024)
    assert valid is False
