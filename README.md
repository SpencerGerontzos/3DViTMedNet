
# 3DViTMedNet: A Hybrid Model for 3D Medical Image Classification

**3DViTMedNet** is a novel deep learning architecture designed for the classification of 3D medical images. This model integrates the strengths of both 3D convolutional neural networks (CNNs) and Vision Transformers (ViTs) to effectively capture local and global spatial relationships in volumetric data such as MRI and CT scans. The repository provides the implementation of the model, data preprocessing scripts, and tools to reproduce the experiments described in our research.

### Key Features:
- **Hybrid Architecture**: Combines 3D CNNs for local feature extraction with Vision Transformers for global feature representation.
- **Data Augmentation Pipeline**: Comprehensive 3D-specific data augmentation techniques, including rotations, scaling, translation, noise addition, and flipping, to improve model robustness and generalization.
- **3D Convolutional Feature Extraction**: Efficiently captures spatial features in volumetric data, preserving critical information across depth, height, and width dimensions.
- **Slice Extraction and Tokenization**: 3D data is divided into 2D slices, processed through a pretrained 2D CNN for tokenization, ensuring computational efficiency while retaining important features.
- **Vision Transformer**: Utilizes the power of the Vision Transformer (ViT) to capture long-range dependencies within the tokenized slices, enhancing the model's ability to classify complex 3D medical data.
- **Modular Design**: The architecture is highly modular, allowing for easy modifications and experimentation with different components.

### Repository Contents:
- **`literature/`**: Contains both thesis and conference submissions.
- **`models/`**: Contains the implementation of the 3DViTMedNet model, including the 3D CNN feature extractor, slice extraction, Vision Transformer, and the final classification head.
- **`data/`**: Includes scripts for data preprocessing, augmentation, and slicing of 3D medical datasets.
- **`training/`**: Training scripts with support for GPU acceleration, model checkpoints, and logging.
- **`evaluation/`**: Tools for evaluating the model's performance on test data, including metrics like accuracy, AUC, and confusion matrices.
- **`experiments/`**: Reproducible configurations for baseline and advanced experiments, with support for various hyperparameter tuning methods.

### How to Use:
1. **Clone the Repository**:
   ```bash
   git clone [repository-link]
   cd 3DViTMedNet
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ and PyTorch installed. Install additional dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**:
   Download the relevant 3D medical dataset (e.g., MRI, CT) and follow the instructions in the `data/` folder to preprocess the data.

4. **Train the Model**:
   Train the model using the provided training scripts:
   ```bash
   python train.py --config configs/config.yaml
   ```

5. **Evaluate the Model**:
   Run evaluations on test data to generate metrics and visualize results:
   ```bash
   python evaluate.py --checkpoint checkpoints/best_model.pth
   ```

### Datasets:
This repository supports various 3D medical image datasets. You can adapt the code to other datasets by adjusting the preprocessing and slicing methods in the `data/` folder.

### Results:
The experiments conducted with **3DViTMedNet** have shown state-of-the-art performance on multiple 3D medical image classification tasks, demonstrating the effectiveness of the hybrid architecture in preserving both local and global features within the data.

### Citation:
If you use **3DViTMedNet** in your research, please cite the following paper:
```plaintext
@article{yourpaper2024,
  title={3DViTMedNet: A Hybrid Model for 3D Medical Image Classification},
  author={Your Name and Co-Authors},
  journal={Journal of Medical Imaging},
  year={2024}
}
```

### License:
This project is licensed under the MIT License. See the `LICENSE` file for more details.

