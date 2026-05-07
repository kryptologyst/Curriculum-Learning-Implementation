"""Main experiment script for curriculum learning."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.curriculum import BasicDifficultyCurriculum, ConfidenceBasedCurriculum, LossBasedCurriculum
from src.data.datasets import CIFAR10Dataset
from src.models.simple_cnn import ResNetCNN, SimpleCNN
from src.train.trainer import StandardTrainer
from src.utils.device import get_device, log_device_info, set_seed
from src.utils.logging import log_model_info, log_safety_disclaimer, setup_logging


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main experiment function.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed for reproducibility
    set_seed(cfg.experiment.seed)
    
    # Setup logging
    logger, writer = setup_logging(
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        use_tensorboard=cfg.logging.use_tensorboard,
        experiment_name=cfg.experiment.name,
    )
    
    # Log safety disclaimer
    log_safety_disclaimer(logger)
    
    # Log device information
    log_device_info(logger)
    
    # Get device
    device = get_device(cfg.experiment.device)
    logger.info(f"Using device: {device}")
    
    # Create dataset
    logger.info("Loading CIFAR-10 dataset...")
    dataset = CIFAR10Dataset(
        root_dir=cfg.data.root_dir,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        download=True,
        normalize=True,
        augmentation=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    dataset_info = dataset.get_dataset_info()
    logger.info(f"Dataset info: {dataset_info}")
    
    # Create model
    logger.info("Creating model...")
    if cfg.model._target_ == "src.models.simple_cnn.SimpleCNN":
        model = SimpleCNN(
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            hidden_dim=cfg.model.hidden_dim,
        )
    elif cfg.model._target_ == "src.models.simple_cnn.ResNetCNN":
        model = ResNetCNN(
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            hidden_dim=cfg.model.hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model._target_}")
    
    log_model_info(logger, model)
    
    # Create curriculum strategy
    logger.info("Creating curriculum strategy...")
    if cfg.curriculum._target_ == "src.data.curriculum.BasicDifficultyCurriculum":
        curriculum_strategy = BasicDifficultyCurriculum(
            strategy=cfg.curriculum.strategy,
            num_stages=cfg.curriculum.num_stages,
            stage_epochs=cfg.curriculum.stage_epochs,
            difficulty_schedule=cfg.curriculum.difficulty_schedule,
            noise_levels=cfg.curriculum.noise_levels,
            occlusion_sizes=cfg.curriculum.occlusion_sizes,
        )
    elif cfg.curriculum._target_ == "src.data.curriculum.ConfidenceBasedCurriculum":
        curriculum_strategy = ConfidenceBasedCurriculum(
            num_stages=cfg.curriculum.num_stages,
            confidence_thresholds=cfg.curriculum.confidence_thresholds,
            adaptation_rate=cfg.curriculum.adaptation_rate,
        )
    elif cfg.curriculum._target_ == "src.data.curriculum.LossBasedCurriculum":
        curriculum_strategy = LossBasedCurriculum(
            num_stages=cfg.curriculum.num_stages,
            loss_thresholds=cfg.curriculum.loss_thresholds,
            window_size=cfg.curriculum.window_size,
        )
    else:
        raise ValueError(f"Unknown curriculum strategy: {cfg.curriculum._target_}")
    
    logger.info(f"Curriculum strategy: {curriculum_strategy.__class__.__name__}")
    logger.info(f"Number of stages: {curriculum_strategy.num_stages}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = StandardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        curriculum_strategy=curriculum_strategy,
        device=device,
        optimizer=cfg.trainer.optimizer,
        scheduler=cfg.trainer.scheduler,
        loss_function=cfg.trainer.loss_function,
        metrics=cfg.trainer.metrics,
        checkpoint_dir="assets/checkpoints",
        log_dir="assets/logs",
        experiment_name=cfg.experiment.name,
    )
    
    # Train model
    logger.info("Starting training...")
    training_results = trainer.train(
        epochs=cfg.training.epochs,
        early_stopping_patience=cfg.training.early_stopping.patience,
        save_checkpoints=True,
    )
    
    # Log final results
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {training_results['best_val_accuracy']:.4f}")
    logger.info(f"Final test accuracy: {training_results['final_test_accuracy']:.4f}")
    logger.info(f"Final test F1: {training_results['final_test_f1']:.4f}")
    
    # Save results
    results_path = Path("assets/results")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = results_path / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    
    # Save training results
    results_file = results_path / "training_results.yaml"
    OmegaConf.save(training_results, results_file)
    
    logger.info(f"Results saved to {results_path}")
    
    # Log curriculum analysis
    if curriculum_strategy is not None:
        logger.info("Curriculum Learning Analysis:")
        for stage in range(curriculum_strategy.num_stages):
            stage_info = curriculum_strategy.get_stage_info(stage)
            logger.info(f"  Stage {stage}: {stage_info}")


def run_baseline_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run baseline experiment without curriculum learning.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Training results
    """
    # Set random seed
    set_seed(cfg.experiment.seed)
    
    # Setup logging
    logger, _ = setup_logging(
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        use_tensorboard=False,
        experiment_name=f"{cfg.experiment.name}_baseline",
    )
    
    logger.info("Running baseline experiment (no curriculum learning)")
    
    # Get device
    device = get_device(cfg.experiment.device)
    
    # Create dataset
    dataset = CIFAR10Dataset(
        root_dir=cfg.data.root_dir,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        download=True,
        normalize=True,
        augmentation=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    
    # Create model
    model = SimpleCNN(
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        hidden_dim=cfg.model.hidden_dim,
    )
    
    # Create trainer without curriculum strategy
    trainer = StandardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        curriculum_strategy=None,  # No curriculum learning
        device=device,
        optimizer=cfg.trainer.optimizer,
        scheduler=cfg.trainer.scheduler,
        loss_function=cfg.trainer.loss_function,
        metrics=cfg.trainer.metrics,
        checkpoint_dir="assets/checkpoints",
        log_dir="assets/logs",
        experiment_name=f"{cfg.experiment.name}_baseline",
    )
    
    # Train model
    training_results = trainer.train(
        epochs=cfg.training.epochs,
        early_stopping_patience=cfg.training.early_stopping.patience,
        save_checkpoints=True,
    )
    
    logger.info(f"Baseline results - Test accuracy: {training_results['final_test_accuracy']:.4f}")
    
    return training_results


def run_curriculum_comparison(cfg: DictConfig) -> None:
    """
    Run comparison between different curriculum learning strategies.
    
    Args:
        cfg: Configuration object
    """
    logger, _ = setup_logging(
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        use_tensorboard=False,
        experiment_name=f"{cfg.experiment.name}_comparison",
    )
    
    logger.info("Running curriculum learning comparison")
    
    # Strategies to compare
    strategies = [
        ("baseline", None),
        ("basic_difficulty", BasicDifficultyCurriculum()),
        ("confidence_based", ConfidenceBasedCurriculum()),
        ("loss_based", LossBasedCurriculum()),
    ]
    
    results = {}
    
    for strategy_name, strategy in strategies:
        logger.info(f"Running experiment: {strategy_name}")
        
        # Update config for this strategy
        cfg_copy = cfg.copy()
        cfg_copy.experiment.name = f"{cfg.experiment.name}_{strategy_name}"
        
        if strategy is None:
            # Baseline experiment
            result = run_baseline_experiment(cfg_copy)
        else:
            # Curriculum experiment
            cfg_copy.curriculum._target_ = strategy.__class__.__module__ + "." + strategy.__class__.__name__
            result = main(cfg_copy)
        
        results[strategy_name] = result
    
    # Log comparison results
    logger.info("Curriculum Learning Comparison Results:")
    for strategy_name, result in results.items():
        if isinstance(result, dict):
            test_acc = result.get("final_test_accuracy", 0.0)
            logger.info(f"  {strategy_name}: {test_acc:.4f}")
        else:
            logger.info(f"  {strategy_name}: Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Learning Experiment")
    parser.add_argument("--mode", choices=["train", "baseline", "compare"], default="train",
                       help="Experiment mode")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main()
    elif args.mode == "baseline":
        # Run baseline experiment
        from hydra import initialize, compose
        with initialize(config_path="configs"):
            cfg = compose(config_name="config")
        run_baseline_experiment(cfg)
    elif args.mode == "compare":
        # Run comparison
        from hydra import initialize, compose
        with initialize(config_path="configs"):
            cfg = compose(config_name="config")
        run_curriculum_comparison(cfg)
