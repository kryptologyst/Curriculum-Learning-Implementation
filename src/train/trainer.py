"""Training system for curriculum learning."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.metrics.evaluation import ModelEvaluator
from src.utils.device import get_device
from src.utils.logging import log_training_progress, log_curriculum_stage


class StandardTrainer:
    """
    Standard trainer for curriculum learning experiments.
    
    Handles training loops, validation, checkpointing, and curriculum
    stage management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        curriculum_strategy: Optional[Any] = None,
        device: Optional[str] = None,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        loss_function: str = "cross_entropy",
        metrics: Optional[List[str]] = None,
        checkpoint_dir: str = "assets/checkpoints",
        log_dir: str = "assets/logs",
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            curriculum_strategy: Curriculum learning strategy
            device: Device to train on
            optimizer: Optimizer type
            scheduler: Learning rate scheduler type
            loss_function: Loss function type
            metrics: List of metrics to compute
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.curriculum_strategy = curriculum_strategy
        self.device = get_device(device)
        self.metrics = metrics or ["accuracy", "f1", "auroc"]
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer)
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler)
        
        # Initialize loss function
        self.criterion = self._create_loss_function(loss_function)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            model=self.model,
            device=self.device,
            num_classes=10,  # CIFAR-10
            class_names=[
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        )
        
        # Setup logging
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.experiment_name = experiment_name or "curriculum_experiment"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = []
    
    def _create_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """Create optimizer based on type."""
        if optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self, scheduler_type: str) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on type."""
        if scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        elif scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        elif scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        else:
            return None
    
    def _create_loss_function(self, loss_type: str) -> nn.Module:
        """Create loss function based on type."""
        if loss_type.lower() == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_type.lower() == "focal":
            # Simple focal loss implementation
            return FocalLoss()
        else:
            raise ValueError(f"Unknown loss function type: {loss_type}")
    
    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Get current curriculum stage
        stage = 0
        if self.curriculum_strategy is not None:
            stage = self.curriculum_strategy.get_stage(epoch, total_epochs)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Apply curriculum modifications
            if self.curriculum_strategy is not None:
                batch = (inputs, targets)
                modified_batch = self.curriculum_strategy.apply_curriculum(batch, stage)
                inputs, targets = modified_batch
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "curriculum_stage": stage,
        }
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing validation metrics
        """
        val_metrics = self.evaluator.evaluate(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar("Validation/Loss", val_metrics["avg_loss"], epoch)
        self.writer.add_scalar("Validation/Accuracy", val_metrics["accuracy"], epoch)
        self.writer.add_scalar("Validation/F1_Macro", val_metrics["f1_macro"], epoch)
        
        return val_metrics
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 10,
        save_checkpoints: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_checkpoints: Whether to save model checkpoints
            
        Returns:
            Dictionary containing training results
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch, epochs)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["accuracy"])
                else:
                    self.scheduler.step()
            
            # Log progress
            log_training_progress(
                logger=None,  # Would need to pass logger
                epoch=epoch + 1,
                total_epochs=epochs,
                train_loss=train_metrics["train_loss"],
                val_loss=val_metrics["avg_loss"],
                train_acc=train_metrics["train_accuracy"],
                val_acc=val_metrics["accuracy"],
                stage=train_metrics.get("curriculum_stage", 0),
            )
            
            # Log curriculum stage information
            if self.curriculum_strategy is not None:
                stage = train_metrics["curriculum_stage"]
                stage_info = self.curriculum_strategy.get_stage_info(stage)
                log_curriculum_stage(
                    logger=None,  # Would need to pass logger
                    stage=stage + 1,
                    total_stages=self.curriculum_strategy.num_stages,
                    stage_info=stage_info,
                )
            
            # Save training history
            epoch_history = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["train_loss"],
                "train_accuracy": train_metrics["train_accuracy"],
                "val_loss": val_metrics["avg_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "curriculum_stage": train_metrics.get("curriculum_stage", 0),
            }
            self.training_history.append(epoch_history)
            
            # Log to TensorBoard
            self.writer.add_scalar("Train/Loss", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("Train/Accuracy", train_metrics["train_accuracy"], epoch)
            self.writer.add_scalar("Train/Curriculum_Stage", train_metrics.get("curriculum_stage", 0), epoch)
            
            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                patience_counter = 0
                
                if save_checkpoints:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Final evaluation
        test_metrics = self.evaluator.evaluate(self.test_loader)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return {
            "best_val_accuracy": best_val_acc,
            "final_test_accuracy": test_metrics["accuracy"],
            "final_test_f1": test_metrics["f1_macro"],
            "training_history": self.training_history,
            "test_metrics": test_metrics,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "training_history": self.training_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.best_val_acc = checkpoint["best_val_acc"]
        self.training_history = checkpoint.get("training_history", [])
        
        print(f"Loaded checkpoint from {checkpoint_path}")


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions
            targets: True labels
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
