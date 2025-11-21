https://github.com/vidhinainwal/collabrative_cnn_teamVC
Pull Requests- https://github.com/vidhinainwal/collabrative_cnn_teamVC/pull/1
Pull Requests- https://github.com/vidhinainwal/collabrative_cnn_teamVC/pull/2

Fork - https://github.com/vidhinainwal/collabrative_cnn_teamVC/tree/dev_user1

– Base models used by both users

Model v1 – Custom CNN (User 1)
Model v1 is a custom-built Convolutional Neural Network (CNN) designed for binary classification (Cats vs Dogs). It was intentionally kept lightweight to ensure fast training and easy experimentation.

Architecture Summary
Conv2D(64 filters, 3×3 kernel, ReLU)
Extracts low-level image patterns (edges, textures).
MaxPooling2D
Reduces spatial size and computation.
Conv2D(128 filters, 3×3, ReLU)
Learns more complex patterns (fur, shapes).
MaxPooling2D
Flatten
Dense(128, ReLU), Dense(64, ReLU), Dense(32, ReLU)
Fully connected layers learn object-level features.
Dense(1, Sigmoid)
Outputs probability of class (0 = cat, 1 = dog).

Key Characteristics
Small, simple CNN.
Works well on small datasets.
Fast to train (good for Model v1 baseline).
Limited representational power compared to modern pretrained models.


Model v2 – Transfer Learning with ResNet50 (User 2)
Model v2 uses transfer learning with a pretrained ResNet50 backbone.
ResNet50 is a deep convolutional architecture with 50 layers and residual connections, originally trained on ImageNet (1.2M images, 1000 classes).

Architecture Summary
ResNet50 (include_top=False, pooling='avg')
The base model extracts powerful high-level features.
Freeze all layers except last 10 → allows fine-tuning on Cats vs Dogs.
Dropout(0.5)
Reduces overfitting.
Dense(256, ReLU)
Learns task-specific patterns.
Dropout(0.3)
Dense(NUM_CLASSES = 2, Softmax)
Outputs cat/dog probabilities.

Key Characteristics
Very strong feature extractor compared to a small CNN.
Better generalization, higher accuracy, better robustness.
Requires more training time and GPU due to deeper architecture.
Ideal choice for Model v2 improvements.



– Dataset descriptions
User 1-
Cat and Dog

User 2
Dogs vs. Cats Redux: Kernels Edition
We are excited to bring back the infamous Dogs vs. Cats classification problem as a playground competition with kernels enabled. Although modern techniques may make light of this once-difficult problem, it is through practice of new techniques on old datasets that we will make light of machine learning's future challenges.


– Metrics on both datasets
– Observations on generalization and domain shift
