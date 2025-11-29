# Face Detection using YOLOv8

A custom-trained YOLOv8 face detection model achieving high accuracy across various face orientations and conditions.

## 📋 Project Overview

This project implements a face detection system using YOLOv8n (nano), trained on a custom annotated dataset from Roboflow. Face detection is a fundamental computer vision task with applications in biometrics, security systems, photo tagging, and human-computer interaction.

## 🎯 Objectives

- Train a custom YOLOv8 model for accurate face detection
- Evaluate performance using standard object detection metrics
- Create a robust model capable of detecting faces in various conditions

## 📊 Dataset

**Source**: [Roboflow Face Detection Dataset](https://app.roboflow.com/bishes/face-detection-mik1i-8e409/models/face-detection-mik1i-8e409/2)

The dataset consists of images with bounding box annotations around human faces, including various:

- Face orientations
- Lighting conditions
- Occlusion levels
- Background complexities

**Dataset Splits**:

- Training set
- Validation set
- Test set

## 🏗️ Model Architecture & Training

**Model**: YOLOv8n (nano) - The smallest and fastest YOLOv8 variant

**Training Configuration**:

- **Epochs**: 25
- **Batch Size**: 16
- **Image Size**: 640×640
- **Optimizer & Loss**: Default YOLOv8 settings
- **Environment**: Python with Ultralytics library

## 📈 Results

### Training Results

| Metric       | Score |
| ------------ | ----- |
| Precision    | 0.97  |
| Recall       | 0.95  |
| mAP@0.5      | 0.98  |
| mAP@0.5:0.95 | 0.75  |

### Validation Results

| Metric       | Score |
| ------------ | ----- |
| Precision    | 0.97  |
| Recall       | 0.95  |
| mAP@0.5      | 0.98  |
| mAP@0.5:0.95 | 0.75  |

### Test Results

| Metric       | Score |
| ------------ | ----- |
| Precision    | 0.92  |
| Recall       | 0.905 |
| mAP@0.5      | 0.963 |
| mAP@0.5:0.95 | 0.68  |

**Key Findings**:

- High precision indicates minimal false positives
- Strong recall demonstrates effective detection of actual faces
- Excellent mAP scores show balanced precision-recall performance

## 📁 Repository Structure

```
.
├── weights/
│   └── best.pt                    # Trained model weights
├── face-detection/                # Training results and metrics
├── val3/                          # Validation batch images
├── val4/                          # Additional validation results
├── predicted_results/             # Inference predictions
└── README.md
```

**Important Files**:

- `best.pt`: Trained model weights ([location](results/face-detection/weights/best.pt))
- Training metrics and evaluation results in `face-detection/`, `val3/`, `val4/`
- Prediction outputs in `predicted_results/` folder ([here](/results/predict/))

## 🚀 Usage

### Installation

```bash
pip install ultralytics
pip install torch torchvision
```

### Running Inference

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('weights/best.pt')

# Run inference on an image
results = model('path/to/image.jpg')

# Display results
results[0].show()

# Save results
results[0].save('output.jpg')
```

### Batch Prediction

```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')

# Run on multiple images
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

for i, result in enumerate(results):
    result.save(f'output_{i}.jpg')
```

## 💡 Performance Analysis

### Strengths

- High accuracy in detecting faces under normal conditions
- Robust performance across different face orientations
- Fast inference speed due to YOLOv8n architecture
- Minimal false positives (high precision)

### Limitations

The model occasionally struggles with:

- Small faces in the background
- Faces under poor lighting conditions
- Occluded or partially visible faces
- Extreme angles or unusual poses

## 🔮 Future Improvements

- **Extended Training**: Train for more epochs to improve generalization
- **Larger Model Variant**: Use YOLOv8s, YOLOv8m, or YOLOv8l for higher accuracy
- **Dataset Enhancement**: Collect more diverse training data
- **Data Augmentation**: Apply stronger augmentation techniques
- **Hard Negative Mining**: Focus on challenging cases
- **Multi-scale Training**: Better detect faces of varying sizes

## 📊 Evaluation Metrics Explained

- **Precision**: Proportion of correct face predictions out of all predictions
- **Recall**: Proportion of actual faces that were detected
- **mAP@0.5**: Mean Average Precision at IoU threshold of 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds from 0.5 to 0.95

## 🛠️ Technologies Used

- **YOLOv8**: State-of-the-art object detection model
- **Ultralytics**: YOLO implementation and training framework
- **PyTorch**: Deep learning framework
- **Roboflow**: Dataset management and annotation platform

## 📝 License

[Specify your license]

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the model architecture
- [Roboflow](https://roboflow.com/) for the dataset platform

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This project demonstrates the effectiveness of YOLOv8 for face detection tasks and provides valuable insights into training object detection models with custom datasets.
