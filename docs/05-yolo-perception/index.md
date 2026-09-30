---
title: YOLO / Object Detection Curriculum
description: Complete, unabridged curriculum for YOLO & Real-Time Object Detection from Level 1 to Level 14 and 5 milestone projects.
---

# ⚡ YOLO / Object Detection Curriculum

<div class="notion-callout">
  <div class="notion-callout-icon">🎯</div>
  <div class="notion-callout-content">
    <strong>Beginner → Advanced curriculum focused on practical Computer Vision engineering and real-time perception.</strong>
  </div>
</div>

<div class="notion-card-meta" style="margin-bottom: 2rem;">
  <span class="notion-tag notion-tag-yellow">14 Levels</span>
  <span class="notion-tag notion-tag-gray">44 Topics</span>
  <span class="notion-tag notion-tag-red">5 Milestone Projects</span>
</div>

---

## LEVEL 1 — OBJECT DETECTION FUNDAMENTALS

### 1. Object Detection Basics
* Classification vs object detection
* Detection vs segmentation
* Bounding boxes
* Classes
* Confidence scores
* Object localization
* Single-object vs multi-object detection
* Detection pipeline

### 2. Bounding Boxes
* (x, y, width, height)
* (x1, y1, x2, y2)
* Center-based representation
* Normalized coordinates
* Bounding-box conversion
* Bounding-box visualization

### 3. IoU — Intersection over Union
* Intersection
* Union
* IoU calculation
* IoU for detection
* IoU thresholds
* IoU limitations

---

## LEVEL 2 — YOLO FUNDAMENTALS

### 4. What is YOLO?
* You Only Look Once
* One-stage detectors
* Real-time object detection
* YOLO vs two-stage detectors
* YOLO detection pipeline
* Why YOLO is fast

### 5. YOLO Architecture
* Input
* Backbone
* Neck
* Detection head
* Feature extraction
* Multi-scale detection
* Prediction outputs

### 6. YOLO Detection Pipeline
* Image → Resize / Letterbox → Preprocessing → YOLO Model → Raw Predictions → Confidence Filtering → NMS → Final Bounding Boxes

---

## LEVEL 3 — YOLO DATASETS

### 7. Dataset Preparation
* Collecting images
* Image quality
* Class definition
* Dataset structure
* Train / validation / test
* Class distribution
* Dataset leakage
* Data quality checking

### 8. Image Annotation
* Bounding-box annotation
* Manual annotation
* Annotation tools
* Label formats
* YOLO annotation format
* COCO format
* Pascal VOC format

### 9. YOLO Label Format
* class_id x_center y_center width height
* Normalized coordinates
* Class IDs
* One .txt file per image
* Multiple objects per image

### 10. Dataset Configuration
* Dataset YAML
* Training images
* Validation images
* Number of classes
* Class names
* Dataset paths

---

## LEVEL 4 — YOLO TRAINING

### 11. Training Workflow
* Model selection
* Dataset configuration
* Image size
* Batch size
* Epochs
* Training
* Validation
* Checkpoints
* Best model
* Last model

### 12. Transfer Learning
* Pretrained YOLO models
* Starting from pretrained weights
* Fine-tuning
* Frozen layers
* Full fine-tuning
* Learning rate

### 13. YOLO Training Parameters
* Epochs
* Batch size
* Image size
* Learning rate
* Weight decay
* Optimizer
* Warmup
* Augmentation parameters

### 14. Data Augmentation
* Horizontal flip
* Vertical flip
* Rotation
* Scaling
* Translation
* Cropping
* Color augmentation
* Mosaic augmentation
* MixUp

---

## LEVEL 5 — YOLO LOSS & TRAINING MECHANICS

### 15. Detection Loss
* Bounding-box loss
* Classification loss
* Objectness/confidence loss
* Total detection loss

### 16. Bounding-Box Losses
* IoU loss
* GIoU
* DIoU
* CIoU
* Why IoU-based losses are used

### 17. Classification Loss
* Binary classification
* Multi-class classification
* BCE
* Class probabilities

### 18. Confidence & Objectness
* Object confidence
* Class confidence
* Final detection confidence
* Confidence threshold

---

## LEVEL 6 — POST-PROCESSING

### 19. Non-Maximum Suppression — NMS
* Why duplicate detections happen
* Confidence filtering
* IoU threshold
* NMS algorithm
* Class-aware NMS
* Class-agnostic NMS

### 20. Detection Thresholds
* Confidence threshold
* IoU threshold
* Precision vs recall tradeoff
* Threshold tuning

---

## LEVEL 7 — YOLO EVALUATION

### 21. Detection Metrics
* Precision
* Recall
* F1 score
* Confusion matrix
* True Positive
* False Positive
* False Negative

### 22. mAP
* AP
* mAP
* mAP@50
* mAP@50:95
* Precision-recall curve
* Per-class AP

### 23. Detection Error Analysis
* Missed objects
* False detections
* Wrong class
* Poor localization
* Small-object failures
* Occlusion
* Difficult backgrounds

---

## LEVEL 8 — MODERN YOLO WORKFLOW

### 24. Ultralytics YOLO
* Installation
* CLI
* Python API
* Model loading
* Prediction
* Training
* Validation
* Export

### 25. Inference
* Image inference
* Video inference
* Webcam inference
* Batch inference
* Confidence threshold
* Visualization
* Saving results

### 26. Custom Object Detection
* Collect Dataset
* Annotate
* Prepare YOLO Dataset
* Train
* Validate
* Analyze Errors
* Improve Dataset
* Retrain
* Export
* Deploy

---

## LEVEL 9 — YOLO ADVANCED DETECTION

### 27. Small Object Detection
* Why small objects are difficult
* Resolution
* Feature maps
* Multi-scale detection
* Image tiling
* SAHI-style inference

### 28. Crowded Object Detection
* Overlapping objects
* Occlusion
* NMS limitations
* Confidence tuning
* Dataset quality

### 29. Custom Training Improvements
* Dataset balancing
* Augmentation tuning
* Image resolution
* Batch-size tuning
* Learning-rate tuning
* Model-size selection
* Error-driven improvement

### 30. Model Variants
* Nano / tiny models
* Small models
* Medium models
* Large models
* Extra-large models
* Accuracy ↔ Speed ↔ Memory ↔ Hardware

---

## LEVEL 10 — YOLO TASKS

### 31. YOLO Classification
* Image classification
* Training
* Validation
* Inference

### 32. YOLO Segmentation
* Instance segmentation
* Masks
* Detection + segmentation
* Mask visualization

### 33. YOLO Pose
* Keypoints
* Human pose
* Skeleton
* Keypoint detection

### 34. YOLO Tracking
* Detection + tracking
* Object IDs
* Video tracking
* Track management
* Integration with tracking algorithms

---

## LEVEL 11 — YOLO + OPENCV

### 35. YOLO + OpenCV
* OpenCV image input
* Preprocessing
* YOLO inference
* Bounding-box drawing
* Video processing
* Webcam detection

### 36. Real-Time Detection
* Camera → OpenCV → Frame → Preprocessing → YOLO → Detection → NMS → Tracking → Visualization
* FPS
* Latency
* Throughput

---

## LEVEL 12 — YOLO DEPLOYMENT

### 37. Model Export
* PyTorch
* ONNX
* TensorRT
* OpenVINO
* CoreML where relevant

### 38. ONNX Deployment
* Export
* ONNX Runtime
* Input/output handling
* CPU inference
* GPU inference

### 39. TensorRT
* TensorRT basics
* FP32
* FP16
* INT8
* Engine creation
* Inference optimization

### 40. Real-Time Optimization
* Input resolution
* Batch size
* FP16
* INT8
* GPU utilization
* Memory transfer
* Preprocessing optimization
* Post-processing optimization
* FPS vs latency

---

## LEVEL 13 — PRODUCTION YOLO

### 41. Production Pipeline
* Camera/video input
* Preprocessing
* Inference
* Post-processing
* Tracking
* Output
* Logging
* Monitoring

### 42. API Deployment
* FastAPI
* Image inference API
* Video inference
* Batch inference
* Docker

### 43. Model Versioning
* Model versions
* Dataset versions
* Configuration
* Experiment tracking
* Reproducibility

### 44. Monitoring
* Inference latency
* FPS
* GPU utilization
* Detection confidence
* Detection distribution
* Model degradation

---

## LEVEL 14 — YOLO PROJECTS

### Beginner — Project 1: Custom Object Detector
* 3–5 classes
* Custom dataset
* Train YOLO
* Evaluate
* Deploy webcam detection

### Intermediate — Project 2: Real-Time Object Detection System
* YOLO + OpenCV
* Webcam
* FPS counter
* Detection visualization
* Configurable confidence

### Intermediate+ — Project 3: Multi-Object Detection + Tracking
* YOLO
* Tracking
* Object IDs
* Counting
* Entry/exit detection

### Advanced — Project 4: Edge AI Object Detection
* YOLO
* ONNX/TensorRT
* NVIDIA GPU/Jetson
* Real-time optimization
* FPS benchmarking

### Production-Level — Project 5: Real-Time Perception System
* Camera → OpenCV → YOLO Detection → Tracking → Object Classification → Spatial / Distance Information → Event Detection → API / Dashboard

---

## 🎯 Priority for your Computer Vision Career

<div class="notion-callout">
  <div class="notion-callout-icon">📌</div>
  <div class="notion-callout-content">
    <strong>Must master:</strong> Object Detection → Bounding Boxes → IoU → YOLO Architecture → Dataset/Annotation → Training → NMS → mAP → Custom YOLO → OpenCV Integration → Tracking → ONNX/TensorRT → Real-Time Optimization.<br/><br/>
    <strong>Then expand into:</strong> YOLO Segmentation → YOLO Pose → YOLO Tracking → Edge Deployment.
  </div>
</div>
