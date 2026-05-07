"""Logging utilities for the curriculum learning project."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    log_dir: str = "assets/logs",
    level: str = "INFO",
    use_tensorboard: bool = True,
    experiment_name: Optional[str] = None,
) -> tuple[logging.Logger, Optional[SummaryWriter]]:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_tensorboard: Whether to enable TensorBoard logging
        experiment_name: Name of the experiment for TensorBoard
        
    Returns:
        tuple: Logger instance and optional TensorBoard writer
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"experiment_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    
    logger = logging.getLogger("curriculum_learning")
    
    # Set up TensorBoard
    writer = None
    if use_tensorboard:
        tb_dir = log_path / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        
        if experiment_name:
            tb_dir = tb_dir / experiment_name
        
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logging enabled: {tb_dir}")
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, writer


def log_model_info(logger: logging.Logger, model: torch.nn.Module) -> None:
    """
    Log model information.
    
    Args:
        logger: Logger instance
        model: PyTorch model to log information about
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    train_acc: Optional[float] = None,
    val_acc: Optional[float] = None,
    stage: Optional[int] = None,
) -> None:
    """
    Log training progress information.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        val_loss: Validation loss (optional)
        train_acc: Training accuracy (optional)
        val_acc: Validation accuracy (optional)
        stage: Current curriculum stage (optional)
    """
    progress = f"Epoch {epoch}/{total_epochs}"
    if stage is not None:
        progress += f" (Stage {stage})"
    
    progress += f" - Train Loss: {train_loss:.4f}"
    
    if val_loss is not None:
        progress += f", Val Loss: {val_loss:.4f}"
    
    if train_acc is not None:
        progress += f", Train Acc: {train_acc:.4f}"
    
    if val_acc is not None:
        progress += f", Val Acc: {val_acc:.4f}"
    
    logger.info(progress)


def log_curriculum_stage(
    logger: logging.Logger,
    stage: int,
    total_stages: int,
    stage_info: dict,
) -> None:
    """
    Log curriculum stage information.
    
    Args:
        logger: Logger instance
        stage: Current stage number
        total_stages: Total number of stages
        stage_info: Dictionary containing stage-specific information
    """
    logger.info(f"Starting Curriculum Stage {stage}/{total_stages}")
    for key, value in stage_info.items():
        logger.info(f"  {key}: {value}")


def log_evaluation_results(
    logger: logging.Logger,
    results: dict,
    split: str = "test",
) -> None:
    """
    Log evaluation results.
    
    Args:
        logger: Logger instance
        results: Dictionary containing evaluation metrics
        split: Data split name (train, val, test)
    """
    logger.info(f"{split.capitalize()} Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


def log_safety_disclaimer(logger: logging.Logger) -> None:
    """
    Log safety and ethics disclaimer.
    
    Args:
        logger: Logger instance
    """
    logger.warning("=" * 80)
    logger.warning("SAFETY AND ETHICS DISCLAIMER")
    logger.warning("=" * 80)
    logger.warning("This is a research/educational implementation of curriculum learning.")
    logger.warning("NOT FOR PRODUCTION USE OR REAL-WORLD DECISIONS.")
    logger.warning("The models and results are for demonstration purposes only.")
    logger.warning("Always validate results with domain experts before any application.")
    logger.warning("=" * 80)
