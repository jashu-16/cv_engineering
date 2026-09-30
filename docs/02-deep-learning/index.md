---
title: Deep Learning Curriculum
description: Complete, unabridged curriculum for Deep Learning from Level 1 to Level 13.
---

<div class="pixel-banner">
  <div class="pixel-hud-row">
    <div class="pixel-tag"><span class="pixel-indicator"></span>NEURAL_CORE // CURRICULUM_INDEX</div>
    <div class="pixel-meta-right">13 LEVELS // 50 TOPICS</div>
  </div>
  <div class="pixel-main-content">
    <div class="pixel-avatar">🧠</div>
    <div class="pixel-headings">
      <div class="pixel-title-badge">DEEP LEARNING MASTERY CURRICULUM</div>
      <div class="pixel-subtitle">NEURAL NETWORKS • BACKPROP • PYTORCH • CNNS • RESNETS • TRANSFORMERS • DIFFUSION</div>
    </div>
    <div class="pixel-sparklines">
      <div class="pixel-bar" style="--h: 35%"></div>
      <div class="pixel-bar" style="--h: 60%"></div>
      <div class="pixel-bar" style="--h: 85%"></div>
      <div class="pixel-bar" style="--h: 70%"></div>
      <div class="pixel-bar" style="--h: 95%"></div>
      <div class="pixel-bar" style="--h: 100%"></div>
    </div>
  </div>
  <div class="pixel-footer-row">
    <span class="pixel-coord">SYS_ID: #02_DL_CORE // ROADMAP: COMPLETE // ARCHITECTURES: SOTA</span>
    <span class="pixel-status-text">[ ACTIVE ]</span>
  </div>
</div>

# 🧠 Deep Learning Curriculum

<div class="notion-callout">
  <div class="notion-callout-icon">🎯</div>
  <div class="notion-callout-content">
    <strong>Beginner → Advanced | Structured learning roadmap</strong><br/>
    <strong>Goal:</strong> Build a strong practical foundation in neural networks, PyTorch, CNNs, modern architectures, training, optimization, and deployment.
  </div>
</div>

<div class="notion-card-meta" style="margin-bottom: 2rem;">
  <span class="notion-tag notion-tag-purple">13 Levels</span>
  <span class="notion-tag notion-tag-gray">50 Topics</span>
  <span class="notion-tag notion-tag-blue">Core AI Roadmap</span>
</div>

---

## [LEVEL 1 — DEEP LEARNING FUNDAMENTALS](level-01-dl-fundamentals.md) { .level-link }

### 1. Introduction to Deep Learning
* What is Deep Learning?
* Machine Learning vs Deep Learning
* Neural networks
* Artificial neurons
* Biological neuron — basic intuition
* Features and representations
* Forward propagation
* Training vs inference

### 2. Neural Network Fundamentals
* Perceptron
* Single-layer neural network
* Multi-layer neural network
* Input layer
* Hidden layers
* Output layer
* Weights
* Bias
* Parameters
* Activations

### 3. Activation Functions
* Why activation functions are needed
* Linear
* Sigmoid
* Tanh
* ReLU
* Leaky ReLU
* ELU
* GELU
* Softmax

---

## [LEVEL 2 — NEURAL NETWORK TRAINING](level-02-neural-network-training.md) { .level-link }

### 4. Forward Propagation
* Layer computation
* Matrix multiplication
* Activation
* Network output
* Computational graph

### 5. Loss Functions
* What is a loss function?
* MSE
* MAE
* Binary Cross Entropy
* Categorical Cross Entropy
* Cross Entropy
* Loss calculation

### 6. Backpropagation
* Why backpropagation
* Computational graphs
* Chain rule
* Gradients
* Gradient calculation
* Weight updates

### 7. Gradient Descent
* Batch gradient descent
* Stochastic gradient descent
* Mini-batch gradient descent
* Learning rate
* Learning-rate schedules

### 8. Optimizers
* SGD
* Momentum
* AdaGrad
* RMSProp
* Adam
* AdamW

---

## [LEVEL 3 — BUILDING NEURAL NETWORKS](level-03-building-neural-networks.md) { .level-link }

### 9. Neural Network Architecture
* Fully connected layers
* Hidden layers
* Depth vs width
* Parameters
* Model capacity

### 10. Training Process
* Epoch
* Batch
* Iteration
* Training loop
* Validation loop
* Checkpoints
* Model saving/loading

### 11. Weight Initialization
* Zero initialization
* Random initialization
* Xavier / Glorot
* He initialization
* Initialization problems

### 12. Normalization
* Batch Normalization
* Layer Normalization
* Instance Normalization
* Group Normalization
* When normalization is useful

---

## [LEVEL 4 — GENERALIZATION & REGULARIZATION](level-04-generalization-regularization.md) { .level-link }

### 13. Overfitting
* Underfitting
* Overfitting
* Generalization
* Model capacity

### 14. Regularization
* L1
* L2 / Weight decay
* Dropout
* Early stopping
* Data augmentation
* Label smoothing

### 15. Training Diagnostics
* Training loss curves
* Validation loss curves
* Accuracy curves
* Learning-rate behavior
* Detecting overfitting
* Detecting underfitting
* Error analysis

---

## [LEVEL 5 — PYTORCH](level-05-pytorch.md) { .level-link }

### 16. PyTorch Fundamentals
* Installation
* Tensors
* Tensor shapes
* Tensor indexing
* Tensor operations
* Broadcasting
* CPU vs GPU

### 17. Autograd
* Automatic differentiation
* Computational graphs
* requires_grad
* backward()
* Gradients

### 18. Neural Network API
* nn.Module
* nn.Linear
* Activation layers
* Sequential models
* Custom models

### 19. Dataset & DataLoader
* Dataset
* DataLoader
* Batch processing
* Shuffling
* Custom datasets
* Transforms

### 20. Training in PyTorch
* Training loop
* Validation loop
* Optimizers
* Loss functions
* Backpropagation
* Checkpoints
* GPU training

### 21. PyTorch Model Management
* state_dict
* Saving/loading models
* Checkpointing
* Evaluation mode
* Inference mode

---

## [LEVEL 6 — CONVOLUTIONAL NEURAL NETWORKS](level-06-cnns.md) { .level-link }

### 22. CNN Fundamentals
* Why CNNs
* Convolution
* Kernel/filter
* Stride
* Padding
* Feature maps
* Receptive field

### 23. Pooling
* Max pooling
* Average pooling
* Global average pooling

### 24. CNN Architecture
* Convolution blocks
* Stacking convolution layers
* Channels
* Spatial dimensions
* Feature hierarchy

### 25. Important CNN Architectures
* LeNet
* AlexNet
* VGG
* GoogLeNet / Inception
* ResNet
* DenseNet
* MobileNet
* EfficientNet

---

## [LEVEL 7 — TRANSFER LEARNING](level-07-transfer-learning.md) { .level-link }

### 26. Transfer Learning
* Pretrained models
* Feature extraction
* Fine-tuning
* Freezing layers
* Unfreezing layers
* Learning-rate selection

### 27. Model Fine-Tuning
* Full fine-tuning
* Partial fine-tuning
* Differential learning rates
* Fine-tuning strategies

---

## [LEVEL 8 — DATA & TRAINING TECHNIQUES](level-08-data-training-techniques.md) { .level-link }

### 28. Data Augmentation
* Why augmentation
* Random crop
* Flip
* Rotation
* Translation
* Scaling
* Color transformations
* Random erasing
* MixUp
* CutMix

### 29. Dataset Preparation
* Dataset organization
* Train/validation/test datasets
* Annotation formats
* Class distribution
* Data quality

### 30. Advanced Training Techniques
* Learning-rate scheduling
* Warmup
* Gradient clipping
* Gradient accumulation
* Mixed precision
* Automatic Mixed Precision

---

## [LEVEL 9 — IMAGE CLASSIFICATION](level-09-image-classification.md) { .level-link }

### 31. Classification Pipeline
* Dataset
* Preprocessing
* Model
* Training
* Validation
* Testing
* Inference

### 32. Classification Metrics
* Accuracy
* Precision
* Recall
* F1-score
* Top-1 accuracy
* Top-5 accuracy
* Confusion matrix

### 33. Classification Improvements
* Class imbalance
* Weighted loss
* Augmentation
* Fine-tuning
* Error analysis

---

## [LEVEL 10 — ADVANCED ARCHITECTURES](level-10-advanced-architectures.md) { .level-link }

### 34. Residual Networks
* Residual learning
* Skip connections
* ResNet blocks
* Why residual connections work

### 35. Attention
* Attention concept
* Query
* Key
* Value
* Self-attention
* Multi-head attention

### 36. Transformers
* Transformer architecture
* Encoder
* Decoder
* Positional encoding
* Self-attention
* Multi-head attention

### 37. Vision Transformers
* Patch embeddings
* Class token
* Position embeddings
* ViT architecture
* Training ViTs

### 38. Modern Architectures
* Swin Transformer
* ConvNeXt
* Efficient architectures
* Hybrid CNN/Transformer architectures

---

## [LEVEL 11 — GENERATIVE DEEP LEARNING](level-11-generative-dl.md) { .level-link }

### 39. Autoencoders
* Encoder
* Latent representation
* Decoder
* Reconstruction loss
* Denoising autoencoder

### 40. Variational Autoencoders
* Latent distributions
* Encoder/decoder
* KL divergence
* Reconstruction loss

### 41. GANs
* Generator
* Discriminator
* Adversarial training
* DCGAN
* Training challenges

### 42. Diffusion Models
* Forward diffusion
* Noise
* Reverse denoising
* U-Net
* Conditioning
* Diffusion model fundamentals

---

## [LEVEL 12 — DEEP LEARNING ENGINEERING](level-12-dl-engineering.md) { .level-link }

### 43. GPU Computing
* CUDA basics
* GPU memory
* CPU vs GPU
* Memory transfer
* Batch size
* GPU utilization

### 44. Performance Optimization
* Mixed precision
* Gradient accumulation
* Efficient DataLoader
* Batch optimization
* Memory optimization

### 45. Model Optimization
* Quantization
* Pruning
* Knowledge distillation
* Model compression

### 46. Model Export
* TorchScript
* ONNX
* Model serialization
* Inference optimization

---

## [LEVEL 13 — EXPERIMENTATION & PRODUCTION](level-13-experimentation-production.md) { .level-link }

### 47. Experiment Tracking
* Experiment organization
* Parameters
* Metrics
* Checkpoints
* MLflow / Weights & Biases

### 48. Reproducibility
* Random seeds
* Dataset versions
* Configuration management
* Environment management

### 49. Model Evaluation & Error Analysis
* Evaluation datasets
* Failure cases
* Confidence analysis
* Misclassification analysis
* Model comparison

### 50. Deployment Fundamentals
* Batch inference
* Real-time inference
* Model serving
* API inference
* Docker
* Monitoring
