"""Utility functions for device management and reproducibility."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ('auto', 'cuda', 'mps', 'cpu', or None)
        
    Returns:
        torch.device: The selected device
        
    Raises:
        RuntimeError: If CUDA is requested but not available
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but not available")
    
    return torch.device(device)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        dict: Device information including CUDA, MPS availability and specs
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved()
    
    return info


def log_device_info(logger) -> None:
    """
    Log device information to the provided logger.
    
    Args:
        logger: Logger instance to use for logging
    """
    device_info = get_device_info()
    
    logger.info("Device Information:")
    logger.info(f"  CUDA Available: {device_info['cuda_available']}")
    logger.info(f"  MPS Available: {device_info['mps_available']}")
    logger.info(f"  CPU Count: {device_info['cpu_count']}")
    
    if device_info["cuda_available"]:
        logger.info(f"  CUDA Device Count: {device_info['cuda_device_count']}")
        logger.info(f"  CUDA Current Device: {device_info['cuda_current_device']}")
        logger.info(f"  CUDA Device Name: {device_info['cuda_device_name']}")
        logger.info(f"  CUDA Memory Allocated: {device_info['cuda_memory_allocated']} bytes")
        logger.info(f"  CUDA Memory Reserved: {device_info['cuda_memory_reserved']} bytes")
