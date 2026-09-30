---
title: OpenCV Curriculum
description: Complete, unabridged curriculum for OpenCV from Level 1 to Level 15.
---

# 👁️ OpenCV Curriculum

<div class="notion-callout">
  <div class="notion-callout-icon">🎯</div>
  <div class="notion-callout-content">
    <strong>Beginner → Advanced | Structured OpenCV learning roadmap</strong><br/>
    <strong>Goal:</strong> Build strong practical OpenCV skills from image fundamentals and processing through features, video, camera geometry, 3D vision, DNN inference, and performance.
  </div>
</div>

<div class="notion-card-meta" style="margin-bottom: 2rem;">
  <span class="notion-tag notion-tag-green">15 Levels</span>
  <span class="notion-tag notion-tag-gray">55 Topics</span>
  <span class="notion-tag notion-tag-yellow">Computer Vision Roadmap</span>
</div>

---

## LEVEL 1 — OPENCV FUNDAMENTALS

### 1. Introduction to OpenCV
* What is OpenCV?
* OpenCV architecture
* OpenCV modules
* Installing OpenCV
* Python + OpenCV
* cv2
* OpenCV + NumPy
* Image representation

### 2. Images & NumPy
* Pixels
* Image dimensions
* Width, height, channels
* Image data types
* uint8
* Image arrays
* Indexing
* Slicing
* ROI
* Copy vs view

### 3. Reading & Writing Images
* cv2.imread()
* cv2.imwrite()
* cv2.imshow()
* Image paths
* Image properties
* Handling invalid images

### 4. Display & User Interaction
* Windows
* Mouse events
* Keyboard events
* Trackbars
* Drawing callbacks

---

## LEVEL 2 — COLOR & IMAGE MANIPULATION

### 5. Color Spaces
* BGR
* RGB
* Grayscale
* HSV
* LAB
* YCrCb
* Color conversion
* Color masking

### 6. Basic Image Operations
* Resize
* Crop
* Flip
* Rotate
* Translate
* Padding
* Image arithmetic
* Bitwise operations

### 7. Drawing
* Lines
* Rectangles
* Circles
* Polygons
* Text
* Bounding boxes
* Annotations

---

## LEVEL 3 — GEOMETRIC TRANSFORMATIONS

### 8. Affine Transformations
* Translation
* Rotation
* Scaling
* Shearing
* Affine transformation matrix

### 9. Perspective Transformation
* Perspective distortion
* Four-point transformation
* Homography
* getPerspectiveTransform()
* warpPerspective()

### 10. Image Warping
* warpAffine()
* warpPerspective()
* Remapping
* Coordinate transformations

---

## LEVEL 4 — IMAGE PROCESSING

### 11. Image Filtering
* Convolution
* Kernels
* Averaging filter
* Gaussian blur
* Median blur
* Bilateral filtering
* Custom filters

### 12. Image Enhancement
* Brightness
* Contrast
* Histogram
* Histogram equalization
* CLAHE
* Gamma correction

### 13. Thresholding
* Binary thresholding
* Binary inverse
* Truncation
* Adaptive thresholding
* Otsu thresholding

### 14. Morphological Operations
* Structuring elements
* Erosion
* Dilation
* Opening
* Closing
* Morphological gradient
* Top hat
* Black hat

### 15. Edge Detection
* Image gradients
* Sobel
* Scharr
* Laplacian
* Canny
* Edge pipelines

---

## LEVEL 5 — CONTOURS & SHAPE ANALYSIS

### 16. Contours
* What are contours?
* findContours()
* Contour hierarchy
* Contour retrieval modes
* Contour approximation

### 17. Contour Features
* Area
* Perimeter
* Centroid
* Bounding rectangle
* Rotated rectangle
* Minimum enclosing circle
* Convex hull
* Convexity defects

### 18. Shape Analysis
* Polygon approximation
* Shape matching
* Geometric properties
* Object counting
* Connected components

---

## LEVEL 6 — FEATURE DETECTION & DESCRIPTION

### 19. Corner Detection
* Harris Corner Detector
* Shi-Tomasi
* Good Features to Track

### 20. Feature Descriptors
* Feature detection vs description
* SIFT
* ORB
* BRISK
* AKAZE

### 21. Feature Matching
* Brute Force Matcher
* BFMatcher
* FLANN
* Descriptor matching
* Ratio test
* Homography-based matching

---

## LEVEL 7 — VIDEO PROCESSING

### 22. Video Input/Output
* VideoCapture
* VideoWriter
* Video files
* FPS
* Resolution
* Codec basics

### 23. Webcam Processing
* Camera capture
* Real-time frames
* Camera properties
* FPS measurement
* Real-time pipelines

### 24. Background Subtraction
* Background modeling
* MOG
* MOG2
* KNN
* Foreground masks

### 25. Optical Flow
* Motion estimation
* Lucas-Kanade
* Sparse optical flow
* Farneback
* Dense optical flow

---

## LEVEL 8 — OBJECT TRACKING

### 26. OpenCV Tracking
* Tracking vs detection
* Single-object tracking
* Tracker initialization
* Bounding-box tracking

### 27. Tracking Algorithms
* CSRT
* KCF
* MIL
* MOSSE
* MedianFlow

### 28. Multi-Object Tracking
* Detection + tracking
* Track IDs
* Track management
* Association
* Basic SORT concepts

---

## LEVEL 9 — CAMERA & CALIBRATION

### 29. Camera Model
* Pinhole camera model
* Coordinate systems
* Image coordinates
* Camera coordinates
* World coordinates

### 30. Camera Calibration
* Intrinsic parameters
* Extrinsic parameters
* Camera matrix
* Distortion coefficients
* Chessboard calibration
* Calibration images

### 31. Lens Distortion
* Radial distortion
* Tangential distortion
* Undistortion
* undistort()
* initUndistortRectifyMap()

### 32. Pose Estimation
* Perspective-n-Point
* solvePnP()
* Rotation vectors
* Translation vectors
* Rodrigues()
* Camera pose

---

## LEVEL 10 — 3D VISION

### 33. Stereo Vision
* Stereo cameras
* Rectification
* Correspondence
* Disparity
* Depth estimation

### 34. Epipolar Geometry
* Epipolar lines
* Fundamental matrix
* Essential matrix
* Stereo geometry

### 35. Depth
* Disparity-to-depth
* Depth maps
* RGB-D
* Depth visualization

### 36. 3D Reconstruction
* Point clouds
* Triangulation
* 3D coordinates
* triangulatePoints()

---

## LEVEL 11 — ADVANCED OPENCV

### 37. Hough Transform
* Hough Lines
* Probabilistic Hough Lines
* Hough Circles
* Shape detection

### 38. Image Pyramids
* Gaussian pyramid
* Laplacian pyramid
* Multi-scale processing

### 39. Template Matching
* Template matching
* Matching methods
* Thresholding results
* Multi-scale matching

### 40. Watershed & Segmentation
* Watershed algorithm
* Distance transform
* Marker-based segmentation

### 41. GrabCut
* Foreground extraction
* Background modeling
* Interactive segmentation

---

## LEVEL 12 — OCR & DOCUMENT PROCESSING

### 42. Document Processing
* Document detection
* Perspective correction
* Document scanning
* Contour-based document extraction

### 43. OCR Pipeline
* Preprocessing
* Thresholding
* Noise removal
* Text region detection
* OCR integration

### 44. OpenCV + OCR Engines
* Tesseract integration
* EasyOCR integration
* PaddleOCR integration
* OCR post-processing

---

## LEVEL 13 — OPENCV DNN

### 45. OpenCV DNN Module
* cv2.dnn
* Loading pretrained models
* Blob creation
* Forward inference
* Model outputs

### 46. Model Formats
* ONNX
* TensorFlow models
* Caffe
* Darknet
* Model conversion basics

### 47. DNN Inference
* Image classification
* Object detection
* Segmentation
* Preprocessing
* Postprocessing
* Confidence thresholds
* NMS

---

## LEVEL 14 — PERFORMANCE & PRODUCTION

### 48. Performance Optimization
* ROI processing
* Vectorization
* Efficient NumPy usage
* Memory management
* FPS optimization
* Profiling

### 49. Parallel Processing
* Multithreading concepts
* Multiprocessing concepts
* Pipeline optimization
* Producer-consumer pipelines

### 50. GPU Acceleration
* CUDA basics
* OpenCV CUDA module
* GPU vs CPU processing
* GPU memory transfers

### 51. Real-Time Vision Pipelines
* Camera → preprocessing → inference → postprocessing
* Frame buffering
* FPS measurement
* Latency
* Throughput
* Real-time visualization

---

## LEVEL 15 — ADVANCED OPENCV INTEGRATION

### 52. OpenCV + NumPy
* Efficient array operations
* Memory layout
* Vectorization
* Zero-copy concepts

### 53. OpenCV + PyTorch
* Tensor ↔ NumPy
* Image preprocessing
* Model input pipelines
* Model output processing

### 54. OpenCV + ONNX Runtime
* ONNX inference
* Preprocessing
* Postprocessing
* Runtime optimization

### 55. OpenCV + TensorRT
* TensorRT inference
* Engine loading
* Preprocessing
* Postprocessing
* Real-time inference
