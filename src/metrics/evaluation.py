"""Evaluation metrics for curriculum learning."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


class MetricsCalculator:
    """Calculator for various evaluation metrics."""
    
    def __init__(self, num_classes: int = 10) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes for classification
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        loss: Optional[float] = None,
    ) -> None:
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Predicted class labels
            targets: True class labels
            probabilities: Predicted class probabilities
            loss: Loss value for this batch
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.
        
        Returns:
            Dictionary containing computed metrics
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(targets, predictions)
        metrics["f1_macro"] = f1_score(targets, predictions, average="macro")
        metrics["f1_weighted"] = f1_score(targets, predictions, average="weighted")
        
        # Per-class F1 scores
        f1_per_class = f1_score(targets, predictions, average=None)
        for i, f1 in enumerate(f1_per_class):
            metrics[f"f1_class_{i}"] = f1
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics["confusion_matrix"] = cm.tolist()
        
        # ROC AUC (if probabilities available)
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            
            # Multi-class ROC AUC
            if self.num_classes > 2:
                try:
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="macro"
                    )
                    metrics["roc_auc_ovo"] = roc_auc_score(
                        targets, probabilities, multi_class="ovo", average="macro"
                    )
                except ValueError:
                    # Handle case where some classes are missing
                    metrics["roc_auc_ovr"] = 0.0
                    metrics["roc_auc_ovo"] = 0.0
            else:
                # Binary classification
                try:
                    metrics["roc_auc"] = roc_auc_score(targets, probabilities[:, 1])
                except ValueError:
                    metrics["roc_auc"] = 0.0
        
        # Average loss
        if self.losses:
            metrics["avg_loss"] = np.mean(self.losses)
        
        return metrics
    
    def get_classification_report(self, class_names: Optional[List[str]] = None) -> str:
        """
        Get detailed classification report.
        
        Args:
            class_names: Names of classes for the report
            
        Returns:
            Classification report string
        """
        if not self.predictions:
            return "No predictions available"
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return classification_report(
            targets, predictions, target_names=class_names, digits=4
        )


class CurriculumMetrics:
    """Metrics specific to curriculum learning evaluation."""
    
    def __init__(self, num_stages: int = 3) -> None:
        """
        Initialize curriculum metrics.
        
        Args:
            num_stages: Number of curriculum stages
        """
        self.num_stages = num_stages
        self.stage_metrics = {}
        self.stage_transitions = []
        self.reset()
    
    def reset(self) -> None:
        """Reset all curriculum metrics."""
        self.stage_metrics = {i: MetricsCalculator() for i in range(self.num_stages)}
        self.stage_transitions = []
        self.current_stage = 0
    
    def update_stage(
        self,
        stage: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        loss: Optional[float] = None,
    ) -> None:
        """
        Update metrics for a specific curriculum stage.
        
        Args:
            stage: Current curriculum stage
            predictions: Predicted class labels
            targets: True class labels
            probabilities: Predicted class probabilities
            loss: Loss value for this batch
        """
        if stage in self.stage_metrics:
            self.stage_metrics[stage].update(predictions, targets, probabilities, loss)
        
        # Track stage transitions
        if stage != self.current_stage:
            self.stage_transitions.append({
                "from_stage": self.current_stage,
                "to_stage": stage,
                "epoch": len(self.stage_transitions),
            })
            self.current_stage = stage
    
    def compute_stage_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics for all curriculum stages.
        
        Returns:
            Dictionary containing metrics for each stage
        """
        stage_results = {}
        
        for stage, metrics_calc in self.stage_metrics.items():
            stage_results[f"stage_{stage}"] = metrics_calc.compute_metrics()
        
        # Overall metrics across all stages
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        for metrics_calc in self.stage_metrics.values():
            all_predictions.extend(metrics_calc.predictions)
            all_targets.extend(metrics_calc.targets)
            if metrics_calc.probabilities:
                all_probabilities.extend(metrics_calc.probabilities)
        
        if all_predictions:
            overall_metrics = MetricsCalculator()
            overall_metrics.predictions = all_predictions
            overall_metrics.targets = all_targets
            overall_metrics.probabilities = all_probabilities
            stage_results["overall"] = overall_metrics.compute_metrics()
        
        # Stage transition information
        stage_results["stage_transitions"] = self.stage_transitions
        
        return stage_results
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """
        Get curriculum learning progress information.
        
        Returns:
            Dictionary containing curriculum progress metrics
        """
        progress = {
            "num_stages": self.num_stages,
            "current_stage": self.current_stage,
            "stage_transitions": len(self.stage_transitions),
            "stage_info": {},
        }
        
        for stage, metrics_calc in self.stage_metrics.items():
            stage_metrics = metrics_calc.compute_metrics()
            progress["stage_info"][f"stage_{stage}"] = {
                "num_samples": len(metrics_calc.predictions),
                "accuracy": stage_metrics.get("accuracy", 0.0),
                "f1_macro": stage_metrics.get("f1_macro", 0.0),
                "avg_loss": stage_metrics.get("avg_loss", 0.0),
            }
        
        return progress


class ModelEvaluator:
    """Comprehensive model evaluator for curriculum learning."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize model evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to run evaluation on
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        self.metrics_calc = MetricsCalculator(num_classes)
        self.curriculum_metrics = CurriculumMetrics()
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        curriculum_strategy: Optional[Any] = None,
        stage: int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            curriculum_strategy: Curriculum learning strategy
            stage: Current curriculum stage
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        self.metrics_calc.reset()
        
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Apply curriculum modifications if strategy provided
                if curriculum_strategy is not None:
                    batch = (inputs, targets)
                    modified_batch = curriculum_strategy.apply_curriculum(batch, stage)
                    inputs, targets = modified_batch
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Get predictions and probabilities
                _, predictions = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Update metrics
                self.metrics_calc.update(predictions, targets, probabilities, loss.item())
                
                total_loss += loss.item()
        
        # Compute metrics
        metrics = self.metrics_calc.compute_metrics()
        metrics["avg_loss"] = total_loss / len(dataloader)
        
        # Add curriculum-specific metrics
        if curriculum_strategy is not None:
            stage_info = curriculum_strategy.get_stage_info(stage)
            metrics["curriculum_stage"] = stage_info
        
        return metrics
    
    def evaluate_curriculum(
        self,
        dataloader: torch.utils.data.DataLoader,
        curriculum_strategy: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate model across all curriculum stages.
        
        Args:
            dataloader: Data loader for evaluation
            curriculum_strategy: Curriculum learning strategy
            
        Returns:
            Dictionary containing curriculum evaluation results
        """
        self.curriculum_metrics.reset()
        
        # Evaluate on each stage
        for stage in range(curriculum_strategy.num_stages):
            stage_metrics = self.evaluate(dataloader, curriculum_strategy, stage)
            
            # Update curriculum metrics
            self.curriculum_metrics.update_stage(
                stage,
                torch.tensor(stage_metrics.get("predictions", [])),
                torch.tensor(stage_metrics.get("targets", [])),
                torch.tensor(stage_metrics.get("probabilities", [])),
                stage_metrics.get("avg_loss", 0.0),
            )
        
        return self.curriculum_metrics.compute_stage_metrics()
    
    def get_model_confidence(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Analyze model confidence across the dataset.
        
        Args:
            dataloader: Data loader for analysis
            
        Returns:
            Dictionary containing confidence analysis
        """
        self.model.eval()
        confidences = []
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, batch_targets in dataloader:
                inputs = inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get confidence (max probability)
                batch_confidences = torch.max(probabilities, dim=1)[0]
                _, batch_predictions = torch.max(outputs, 1)
                
                confidences.extend(batch_confidences.cpu().numpy())
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Compute confidence statistics
        confidence_stats = {
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "median_confidence": np.median(confidences),
        }
        
        # Confidence by correctness
        correct_mask = predictions == targets
        confidence_stats["mean_confidence_correct"] = np.mean(confidences[correct_mask])
        confidence_stats["mean_confidence_incorrect"] = np.mean(confidences[~correct_mask])
        
        return confidence_stats
