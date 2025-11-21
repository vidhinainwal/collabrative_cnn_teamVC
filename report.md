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
1. Generalization Differences Between Model v1 and Model v2
Model v1 (Custom CNN) – Weaker Generalization

Trained on a smaller, cleaner dataset with limited variation.

Performs well on training/validation splits but struggles when encountering images with:

unusual lighting

different breeds

occlusions

cluttered backgrounds

This shows overfitting to the narrow training domain and difficulty generalizing to more realistic images.

Model v2 (ResNet50 Transfer Learning) – Stronger Generalization

Benefits from ImageNet pretraining, which exposes the model to:

millions of images

varied objects, textures, and shapes

As a result, Model v2 generalizes far better to unseen cats and dogs, even those with:

complex backgrounds

challenging angles

high intra-class variance

Model v2 maintains high F1-score across different subsets, indicating robust, consistent performance.

2. Domain Shift Observations
What is Domain Shift?

Domain shift occurs when the training dataset and the testing dataset do not follow the same distribution, leading to inevitable performance drops.

Domain Shift in Model v1

Model v1 was trained on a simpler, pre-sorted benchmark dataset, where:

cats and dogs are often centered

backgrounds are cleaner

image quality is consistent

When tested on user 2’s dataset (larger, messier, more diverse), Model v1 would likely show:

reduced accuracy

higher misclassification

instability across samples

This highlights sensitivity to distribution changes.

Domain Shift in Model v2

Model v2, trained on a much larger and more diverse dataset, exhibits far less sensitivity to domain shift.

Transfer learning further reduces domain shift impact because pretrained layers already encode:

edges

textures

high-level semantic features

As a result:

Even when faced with new cat/dog images that differ from training data, Model v2 maintains high performance.

Domain mismatch affects it less severely than Model v1.
