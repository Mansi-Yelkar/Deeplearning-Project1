import logging
import sys
from pathlib import Path

def setup_logger(name: str = "safesight", log_level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger for SafeSight AI."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler with clean formatting
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(level)
        
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        c_handler.setFormatter(fmt)
        logger.addHandler(c_handler)
        
        # Optional log to file in outputs/logs
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        f_handler = logging.FileHandler(log_dir / "safesight.log", encoding="utf-8")
        f_handler.setLevel(level)
        f_handler.setFormatter(fmt)
        logger.addHandler(f_handler)
        
    return logger
