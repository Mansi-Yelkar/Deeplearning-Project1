from pathlib import Path
from typing import Tuple, List

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def validate_image_file(file_name: str, file_size_bytes: int, max_mb: int = 25) -> Tuple[bool, str]:
    """Validates uploaded image file extension and size."""
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, f"Unsupported image format: '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
    
    max_bytes = max_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, f"File exceeds maximum allowed size of {max_mb} MB (received {file_size_bytes / (1024*1024):.2f} MB)."
        
    return True, "Valid image"

def validate_video_file(file_name: str, file_size_bytes: int, max_mb: int = 200) -> Tuple[bool, str]:
    """Validates uploaded video file extension and size."""
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return False, f"Unsupported video format: '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        
    max_bytes = max_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, f"File exceeds maximum allowed size of {max_mb} MB (received {file_size_bytes / (1024*1024):.2f} MB)."
        
    return True, "Valid video"
