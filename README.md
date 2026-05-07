# Curriculum Learning Implementation

A research-focused implementation of curriculum learning for image classification with advanced meta-learning techniques.

**Author**: [kryptologyst](https://github.com/kryptologyst)  
**GitHub**: https://github.com/kryptologyst

## ⚠️ Safety and Ethics Disclaimer

**This is a research/educational implementation. NOT FOR PRODUCTION USE OR REAL-WORLD DECISIONS.**

- The models and results are for demonstration purposes only
- Always validate results with domain experts before any application
- This implementation is designed for educational and research purposes
- No production claims or guarantees are made

## Overview

Curriculum learning is a training strategy where models learn from easier examples first, then gradually progress to more difficult ones. This mimics how humans learn and can lead to better performance and faster convergence.

### Key Features

- **Multiple Curriculum Strategies**: Basic difficulty, confidence-based, and loss-based approaches
- **Modern Architecture**: SimpleCNN and ResNetCNN implementations
- **Comprehensive Evaluation**: Multiple metrics and baselines
- **Interactive Demo**: Streamlit-based visualization tool
- **Reproducible Research**: Deterministic seeding and proper logging
- **Safety-First**: Built-in disclaimers and ethical considerations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Curriculum-Learning-Implementation.git
cd Curriculum-Learning-Implementation

# Install dependencies
pip install -r requirements.txt

# Or install with pip
pip install -e .
```

### Basic Usage

```bash
# Train with curriculum learning
python src/experiment.py

# Run baseline experiment (no curriculum)
python src/experiment.py --mode baseline

# Run comparison between strategies
python src/experiment.py --mode compare

# Launch interactive demo
streamlit run demo/streamlit_demo.py
```

## Curriculum Learning Strategies

### 1. Basic Difficulty Curriculum
Progressively increases difficulty by adding noise and occlusion to training examples.

```python
from src.data.curriculum import BasicDifficultyCurriculum

strategy = BasicDifficultyCurriculum(
    num_stages=3,
    noise_levels=[0.0, 0.1, 0.2],
    occlusion_sizes=[0, 5, 10],
    difficulty_schedule="linear"
)
```

### 2. Confidence-Based Curriculum
Uses model confidence to determine difficulty and adapts the curriculum based on current model performance.

```python
from src.data.curriculum import ConfidenceBasedCurriculum

strategy = ConfidenceBasedCurriculum(
    num_stages=3,
    confidence_thresholds=[0.7, 0.5, 0.3],
    adaptation_rate=0.1
)
```

### 3. Loss-Based Curriculum
Uses training loss to determine difficulty and adapts the curriculum based on current model performance.

```python
from src.data.curriculum import LossBasedCurriculum

strategy = LossBasedCurriculum(
    num_stages=3,
    loss_thresholds=[2.0, 1.0, 0.5],
    window_size=10
)
```

## Architecture

### Models

- **SimpleCNN**: Lightweight CNN with batch normalization and dropout
- **ResNetCNN**: ResNet-style architecture for more complex patterns

### Dataset Support

- **CIFAR-10**: Primary dataset with proper train/val/test splits
- **Synthetic Data**: For testing and experimentation
- **Custom Datasets**: Easy to extend for other image classification tasks

## Evaluation Metrics

### Classification Metrics
- Accuracy, F1-Score (macro/weighted)
- ROC-AUC, Precision-Recall curves
- Per-class performance analysis
- Confusion matrices

### Curriculum-Specific Metrics
- Stage-wise performance tracking
- Curriculum transition analysis
- Difficulty progression evaluation
- Confidence analysis

### Baseline Comparisons
- Standard training (no curriculum)
- Different curriculum strategies
- Ablation studies

## Interactive Demo

Launch the interactive Streamlit demo:

```bash
streamlit run demo/streamlit_demo.py
```

### Demo Features
- **Image Upload**: Test your own images
- **Curriculum Visualization**: See how images change across stages
- **Parameter Tuning**: Adjust curriculum parameters interactively
- **Prediction Analysis**: View confidence scores and probabilities
- **Sample Gallery**: Test on CIFAR-10 samples

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model configurations
- `configs/curriculum/`: Curriculum strategy configurations
- `configs/dataset/`: Dataset configurations

### Example Configuration

```yaml
# configs/config.yaml
experiment:
  name: "curriculum_experiment"
  seed: 42
  device: "auto"

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001

curriculum:
  strategy: "difficulty_based"
  num_stages: 3
  stage_epochs: [10, 20, 20]
  difficulty_schedule: "linear"
```

## Research Focus

This implementation is designed for educational and research purposes:

### Educational Value
- Clear, well-documented code
- Multiple curriculum strategies
- Comprehensive evaluation framework
- Interactive visualization tools

### Research Applications
- Curriculum learning algorithm development
- Meta-learning research
- Educational AI systems
- Training strategy optimization

### Extensibility
- Easy to add new curriculum strategies
- Modular architecture
- Configurable parameters
- Multiple model architectures

## Project Structure

```
curriculum-learning-implementation/
├── src/
│   ├── data/
│   │   ├── curriculum.py          # Curriculum learning strategies
│   │   └── datasets.py            # Dataset implementations
│   ├── models/
│   │   └── simple_cnn.py          # Model architectures
│   ├── metrics/
│   │   └── evaluation.py          # Evaluation metrics
│   ├── train/
│   │   └── trainer.py             # Training system
│   ├── utils/
│   │   ├── device.py              # Device management
│   │   └── logging.py             # Logging utilities
│   └── experiment.py              # Main experiment script
├── configs/                       # Configuration files
├── demo/
│   └── streamlit_demo.py          # Interactive demo
├── assets/                        # Output directory
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # Training logs
│   └── results/                   # Experiment results
├── tests/                         # Unit tests
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Experiments

### Running Experiments

```bash
# Basic curriculum learning experiment
python src/experiment.py

# Baseline comparison
python src/experiment.py --mode baseline

# Strategy comparison
python src/experiment.py --mode compare
```

### Expected Results

Typical performance on CIFAR-10:
- **Baseline**: ~70-75% accuracy
- **Curriculum Learning**: ~75-80% accuracy
- **Advanced Strategies**: ~80-85% accuracy

*Note: Results may vary based on hyperparameters and random seeds.*

## Development

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- Black code formatting
- Ruff linting
- MyPy type checking

### Testing
```bash
# Run tests
pytest tests/

# Type checking
mypy src/

# Code formatting
black src/
ruff check src/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## References

1. Bengio, Y., et al. "Curriculum learning." ICML 2009.
2. Weinshall, D., et al. "Curriculum learning by transfer learning." ICML 2018.
3. Soviany, P., et al. "Curriculum learning: A survey." IJCV 2022.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Author**: [kryptologyst](https://github.com/kryptologyst)
- **GitHub**: https://github.com/kryptologyst
- **Research Community**: Thanks to all researchers advancing curriculum learning

# Curriculum-Learning-Implementation
