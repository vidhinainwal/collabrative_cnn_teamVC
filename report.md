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
Dataset Used for Model v1 (User 1)

Source: Kaggle Cat vs Dog dataset (pre-sorted folder structure)

Structure

The dataset is organized in a clean directory format:

training_set/
    cats/
    dogs/
test_set/
    cats/
    dogs/

Characteristics

Type: Supervised classification

Classes: 2 (Cat, Dog)

Labeling Method: Inferred directly from folder names

Typical Image Size: Varies (resized to 256×256 for training)

Dataset Size: Medium (~2000–5000 images, depending on subset used)

Why this dataset?

Simple to load using image_dataset_from_directory()

Ideal for building a baseline model

Minimal preprocessing required

Good for training a custom CNN from scratch

This dataset helped create Model v1, a lightweight CNN that establishes the baseline classification performance.

Dataset Used for Model v2 (User 2)

Source: Dogs vs Cats Redux: Kernels Edition (Kaggle Challenge Dataset)

Structure

This dataset is provided as zipped files:

train.zip
test.zip


After extraction:

train/
    cat.0.jpg
    dog.1.jpg
    ...
test/
    1.jpg
    2.jpg
    ...

Characteristics

Type: Large-scale supervised classification

Classes: Cat, Dog

Labels: Extracted from filenames

"cat.123.jpg" → cat

"dog.456.jpg" → dog

Image Count (Train): ~25,000

Image Count (Test): ~12,500

Image Conditions: Highly diverse (breeds, angles, occlusions)

Resolution: Varies widely; resized to 128×128

Preprocessing Needed:

unzip archives

read with OpenCV

convert BGR→RGB

resize

one-hot encode labels


– Metrics on both datasets
– Observations on generalization and domain shift
