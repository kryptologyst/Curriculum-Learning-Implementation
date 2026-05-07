"""Interactive demo for curriculum learning using Streamlit."""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.curriculum import BasicDifficultyCurriculum
from src.data.datasets import CIFAR10Dataset
from src.models.simple_cnn import SimpleCNN
from src.utils.device import get_device


class CurriculumLearningDemo:
    """Interactive demo for curriculum learning."""
    
    def __init__(self):
        """Initialize the demo."""
        self.device = get_device("cpu")  # Use CPU for demo
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Initialize components
        self.model = None
        self.curriculum_strategy = None
        self.dataset = None
        
        # Load or create model
        self._load_model()
        
        # Load dataset
        self._load_dataset()
    
    def _load_model(self) -> None:
        """Load or create the model."""
        model_path = Path("assets/checkpoints/best_model.pth")
        
        if model_path.exists():
            # Load trained model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = SimpleCNN(num_classes=10, dropout=0.2, hidden_dim=128)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            st.success("✅ Loaded trained model from checkpoint")
        else:
            # Create new model
            self.model = SimpleCNN(num_classes=10, dropout=0.2, hidden_dim=128)
            self.model.eval()
            st.warning("⚠️ No trained model found. Using untrained model for demo.")
    
    def _load_dataset(self) -> None:
        """Load the dataset."""
        try:
            self.dataset = CIFAR10Dataset(
                root_dir="data",
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                download=True,
                normalize=True,
                augmentation=False,  # No augmentation for demo
                batch_size=1,
                num_workers=0,
                pin_memory=False,
            )
            st.success("✅ Dataset loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            self.dataset = None
    
    def create_curriculum_strategy(self, strategy_type: str, **kwargs) -> None:
        """Create curriculum learning strategy."""
        if strategy_type == "Basic Difficulty":
            self.curriculum_strategy = BasicDifficultyCurriculum(
                num_stages=kwargs.get("num_stages", 3),
                noise_levels=kwargs.get("noise_levels", [0.0, 0.1, 0.2]),
                occlusion_sizes=kwargs.get("occlusion_sizes", [0, 5, 10]),
            )
        else:
            self.curriculum_strategy = None
    
    def predict_image(self, image: Image.Image, stage: int = 0) -> Tuple[str, float, List[float]]:
        """
        Predict class for an image.
        
        Args:
            image: PIL Image
            stage: Curriculum stage
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        if self.model is None:
            return "No model loaded", 0.0, []
        
        # Preprocess image
        image = image.convert("RGB").resize((32, 32))
        
        # Convert to tensor
        transform = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((32, 32)),
        )
        
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        # Normalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Apply curriculum modifications
        if self.curriculum_strategy is not None:
            batch = (image_tensor, torch.tensor([0]))  # Dummy target
            modified_batch = self.curriculum_strategy.apply_curriculum(batch, stage)
            image_tensor = modified_batch[0]
        
        # Predict
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            all_probs = probabilities.squeeze().cpu().numpy().tolist()
        
        return predicted_class, confidence_score, all_probs
    
    def visualize_curriculum_stages(self, image: Image.Image) -> None:
        """Visualize how an image changes across curriculum stages."""
        if self.curriculum_strategy is None:
            st.warning("No curriculum strategy selected")
            return
        
        st.subheader("Curriculum Learning Visualization")
        
        # Create subplots for different stages
        fig, axes = plt.subplots(1, self.curriculum_strategy.num_stages + 1, figsize=(15, 3))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        # Processed images for each stage
        for stage in range(self.curriculum_strategy.num_stages):
            # Apply curriculum modifications
            image_tensor = torch.from_numpy(np.array(image.convert("RGB").resize((32, 32)))).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            batch = (image_tensor, torch.tensor([0]))
            modified_batch = self.curriculum_strategy.apply_curriculum(batch, stage)
            modified_image = modified_batch[0].squeeze().permute(1, 2, 0).numpy()
            modified_image = np.clip(modified_image, 0, 1)
            
            axes[stage + 1].imshow(modified_image)
            stage_info = self.curriculum_strategy.get_stage_info(stage)
            axes[stage + 1].set_title(f"Stage {stage}\nNoise: {stage_info['noise_level']:.1f}")
            axes[stage + 1].axis("off")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def plot_prediction_probabilities(self, probabilities: List[float]) -> None:
        """Plot prediction probabilities."""
        fig = px.bar(
            x=self.class_names,
            y=probabilities,
            title="Prediction Probabilities",
            labels={"x": "Class", "y": "Probability"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_curriculum_info(self) -> None:
        """Show curriculum learning information."""
        st.subheader("Curriculum Learning Information")
        
        st.markdown("""
        **Curriculum Learning** is a training strategy where models learn from easier examples first, 
        then gradually progress to more difficult ones. This mimics how humans learn and can lead 
        to better performance and faster convergence.
        
        ### How it works:
        1. **Stage 0**: Clean images (easy)
        2. **Stage 1**: Images with slight noise (medium)
        3. **Stage 2**: Images with noise and occlusion (hard)
        
        ### Benefits:
        - Better generalization
        - Faster convergence
        - More stable training
        - Improved performance on difficult examples
        """)
        
        if self.curriculum_strategy is not None:
            st.subheader("Current Curriculum Strategy")
            
            for stage in range(self.curriculum_strategy.num_stages):
                stage_info = self.curriculum_strategy.get_stage_info(stage)
                
                with st.expander(f"Stage {stage} Details"):
                    st.write(f"**Noise Level**: {stage_info['noise_level']}")
                    st.write(f"**Occlusion Size**: {stage_info['occlusion_size']}")
                    st.write(f"**Difficulty**: {stage_info['difficulty']:.2f}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Curriculum Learning Demo",
        page_icon="🎓",
        layout="wide",
    )
    
    st.title("🎓 Curriculum Learning Interactive Demo")
    st.markdown("**Author**: [kryptologyst](https://github.com/kryptologyst)")
    
    # Safety disclaimer
    st.warning("""
    ⚠️ **SAFETY DISCLAIMER**: This is a research/educational demo. 
    NOT FOR PRODUCTION USE OR REAL-WORLD DECISIONS.
    """)
    
    # Initialize demo
    demo = CurriculumLearningDemo()
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Curriculum strategy selection
    strategy_type = st.sidebar.selectbox(
        "Curriculum Strategy",
        ["None", "Basic Difficulty"],
        help="Select curriculum learning strategy"
    )
    
    if strategy_type == "Basic Difficulty":
        num_stages = st.sidebar.slider("Number of Stages", 2, 5, 3)
        noise_levels = st.sidebar.text_input(
            "Noise Levels (comma-separated)",
            "0.0,0.1,0.2",
            help="Noise levels for each stage"
        )
        occlusion_sizes = st.sidebar.text_input(
            "Occlusion Sizes (comma-separated)",
            "0,5,10",
            help="Occlusion sizes for each stage"
        )
        
        try:
            noise_levels = [float(x.strip()) for x in noise_levels.split(",")]
            occlusion_sizes = [int(x.strip()) for x in occlusion_sizes.split(",")]
            
            demo.create_curriculum_strategy(
                strategy_type,
                num_stages=num_stages,
                noise_levels=noise_levels,
                occlusion_sizes=occlusion_sizes,
            )
        except ValueError:
            st.sidebar.error("Invalid input format. Use comma-separated numbers.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Image Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to test curriculum learning"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Curriculum stage selection
            if demo.curriculum_strategy is not None:
                stage = st.selectbox(
                    "Curriculum Stage",
                    range(demo.curriculum_strategy.num_stages),
                    help="Select curriculum stage to apply"
                )
            else:
                stage = 0
            
            # Predict button
            if st.button("Predict", type="primary"):
                with st.spinner("Predicting..."):
                    predicted_class, confidence, probabilities = demo.predict_image(image, stage)
                    
                    st.success(f"Predicted: **{predicted_class}**")
                    st.info(f"Confidence: **{confidence:.3f}**")
                    
                    # Plot probabilities
                    demo.plot_prediction_probabilities(probabilities)
        
        # Sample images
        st.header("Sample Images")
        if demo.dataset is not None:
            # Get a few sample images
            sample_images = []
            for i, (image, label) in enumerate(demo.dataset.test_loader.dataset):
                if i >= 5:  # Show only 5 samples
                    break
                sample_images.append((image, label))
            
            for i, (image_tensor, label) in enumerate(sample_images):
                # Convert tensor to PIL image
                image_array = image_tensor.permute(1, 2, 0).numpy()
                image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
                
                col_img, col_info = st.columns([2, 1])
                with col_img:
                    st.image(image_pil, caption=f"Sample {i+1}", width=100)
                with col_info:
                    st.write(f"**True Label**: {demo.class_names[label]}")
                    
                    if st.button(f"Predict {i+1}", key=f"predict_{i}"):
                        predicted_class, confidence, probabilities = demo.predict_image(image_pil, stage)
                        st.write(f"**Predicted**: {predicted_class}")
                        st.write(f"**Confidence**: {confidence:.3f}")
    
    with col2:
        st.header("Curriculum Visualization")
        
        if uploaded_file is not None and demo.curriculum_strategy is not None:
            demo.visualize_curriculum_stages(image)
        
        # Curriculum information
        demo.show_curriculum_info()
        
        # Model information
        st.subheader("Model Information")
        if demo.model is not None:
            total_params = sum(p.numel() for p in demo.model.parameters())
            st.write(f"**Total Parameters**: {total_params:,}")
            st.write(f"**Model Type**: SimpleCNN")
            st.write(f"**Classes**: {len(demo.class_names)}")
        
        # Dataset information
        if demo.dataset is not None:
            st.subheader("Dataset Information")
            dataset_info = demo.dataset.get_dataset_info()
            st.write(f"**Dataset**: {dataset_info['name']}")
            st.write(f"**Train Size**: {dataset_info['train_size']:,}")
            st.write(f"**Val Size**: {dataset_info['val_size']:,}")
            st.write(f"**Test Size**: {dataset_info['test_size']:,}")
            st.write(f"**Image Size**: {dataset_info['image_size']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Demo
    
    This interactive demo showcases **Curriculum Learning** for image classification. 
    Curriculum learning trains models on progressively harder examples, similar to how humans learn.
    
    **Key Features:**
    - Upload your own images to test
    - Visualize curriculum stages
    - Compare predictions across stages
    - Interactive parameter tuning
    
    **Research Focus**: This implementation is designed for educational and research purposes.
    """)
    
    st.markdown("**GitHub**: [kryptologyst](https://github.com/kryptologyst)")


if __name__ == "__main__":
    main()
