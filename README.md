
# Architecture Matters, but Loss Function Is the Heart of Deep Learning

## ResNet: Residual Network

ResNet stands for **Residual Network**.

Traditional Convolutional Neural Networks are very effective for image processing because they can learn visual patterns such as edges, textures, shapes, and complex object features. However, as CNNs become deeper, training becomes more difficult. One major reason is the **vanishing gradient problem**.

During backpropagation, gradients are used to update the weights of the network. In very deep networks, these gradients can become extremely small as they move backward through many layers. As a result, the earlier layers receive very weak updates, and the model may stop learning properly.

ResNet solves this problem using **skip connections**.

In a normal CNN, the network tries to directly learn the complete mapping:

```math
H(x)
````

In ResNet, instead of directly learning `H(x)`, the network learns the residual function:

```math
F(x) = H(x) - x
```

Therefore, the final output becomes:

```math
H(x) = F(x) + x
```

Here, `x` is passed directly through the skip connection.

This makes optimization easier because the network does not need to learn the entire transformation from scratch. It only needs to learn the difference between the input and the desired output.

Because of residual learning, ResNet can go much deeper while still training effectively.

---

## Why ResNet Works Well for Image Understanding

ResNet is powerful because it allows the network to learn both simple and complex features.

In the earlier layers, the network learns low-level features such as:

* Edges
* Corners
* Simple textures
* Color patterns

In the deeper layers, it learns high-level features such as:

* Shapes
* Object parts
* Facial structures
* Class-specific patterns

This makes ResNet useful for many computer vision tasks such as image classification, object detection, medical image analysis, and face recognition.

---

## ResNet-50

ResNet-50 is a 50-layer version of ResNet.

It uses **bottleneck residual blocks**, which make the network deep while keeping computation manageable.

A bottleneck block usually follows this structure:

```text
1x1 convolution → 3x3 convolution → 1x1 convolution
```

The first `1x1` convolution reduces the number of channels.

The `3x3` convolution learns spatial features.

The final `1x1` convolution expands the channels again.

This design helps ResNet-50 learn strong features efficiently.

---

## Face Recognition and Embeddings

Face recognition is not only a classification problem. It is mainly an **embedding learning problem**.

A face image is passed through a backbone network such as ResNet-50, and the model produces a numerical vector called an **embedding**.

```text
face image → ResNet-50 → embedding vector
```

The goal of the embedding space is:

```text
same person      → embeddings should be close
different person → embeddings should be far apart
```

So, the heart of face recognition is the quality of the embedding space.

During testing, two face embeddings can be compared using cosine similarity:

```math
similarity = \frac{A \cdot B}{||A|| \, ||B||}
```

If the embeddings are L2-normalized, cosine similarity becomes a simple dot product.

---

## Why Normal Softmax Is Not Enough

A face recognition model can be trained using normal softmax loss, but softmax is mainly designed for classification.

Softmax answers this question:

```text
Is this image classified correctly?
```

However, face recognition needs a stronger condition:

```text
Are same-person embeddings tightly grouped?
Are different-person embeddings clearly separated?
```

Normal softmax does not strongly enforce this structure.

The softmax probability for class `i` is:

```math
p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
```

The cross-entropy loss is:

```math
L = -\log(p_y)
```

where:

* `z_i` is the logit for class `i`
* `C` is the number of classes
* `y` is the correct class

Softmax encourages the model to classify correctly, but it does not explicitly force embeddings of the same person to be very close or embeddings of different people to be far apart.

This is why margin-based loss functions are important in face recognition.

---

## ArcFace

ArcFace improves face recognition by adding an **angular margin**.

Before applying softmax and cross-entropy, ArcFace normalizes the feature embeddings and the class weights. This makes the model focus mainly on the direction of the embedding rather than its magnitude.

After normalization, embeddings lie on a hypersphere.

The model compares the angle between the face embedding and the class center:

```math
\cos(\theta)
```

ArcFace modifies the target class logit using:

```math
\cos(\theta + m)
```

where:

* `θ` is the angle between the embedding and the correct class center
* `m` is the angular margin

The ArcFace loss is:

```math
L = -\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{
e^{s\cos(\theta_{y_i}+m)}
}{
e^{s\cos(\theta_{y_i}+m)} + \sum_{j \ne y_i} e^{s\cos(\theta_j)}
}
```

where:

* `N` is the batch size
* `s` is the scale factor
* `m` is the angular margin
* `y_i` is the correct class
* `θ_{y_i}` is the angle for the correct class

In simple words:

```text
Softmax → be correct
ArcFace → be correct with an angular margin
```

ArcFace improves the embedding space by:

* Reducing intra-class distance
* Increasing inter-class distance
* Making same-person embeddings tighter
* Making different-person embeddings more separated

---

## AdaFace

AdaFace improves on ArcFace by making the margin adaptive based on image quality.

In real-world face recognition, all images do not have the same quality.

Some images may be:

* Clear
* Frontal
* Well-lit
* Sharp

Other images may be:

* Blurry
* Dark
* Occluded
* Side-facing
* Low resolution

ArcFace applies the same margin to all images. However, applying the same strict margin to low-quality images can make training harder.

AdaFace solves this by adapting the margin according to image quality.

The basic idea is:

```text
high-quality image → larger margin
low-quality image  → smaller margin
```

AdaFace uses the feature norm as a quality indicator. A higher feature norm usually means the model has produced a stronger and more confident representation. A lower feature norm may indicate that the image is harder or lower quality.

A simplified adaptive margin idea can be written as:

```math
m_i = m \cdot q_i
```

where:

* `m_i` is the adaptive margin for sample `i`
* `m` is the base margin
* `q_i` is the quality-based adjustment factor

So AdaFace does not treat every image equally. It gives a stronger margin to high-quality images and a softer margin to low-quality images.

This makes the model more robust for real-world face recognition.

---

## Loss Function and Representation Learning

The architecture decides how features are extracted.

The loss function decides what the model is forced to learn.

For example:

* ResNet extracts visual features.
* Softmax trains the model for classification.
* ArcFace shapes the embedding space using angular margin.
* AdaFace improves the margin by considering image quality.

This is why the loss function is extremely important in deep learning, especially in face recognition.

A strong architecture alone is not enough. The model must also be trained with the right objective.

---

# Projects

## Project 1: ResNet-50 + AdaFace Face Recognition

This project implements a face recognition pipeline using **ResNet-50** as the backbone and **AdaFace** as the loss function. Face detection is used to crop faces before training and testing. The model learns embeddings for face images, and cosine similarity is used to compare two faces. The aim of this project is to understand how modern face recognition systems combine residual networks, embedding learning, angular-margin losses, adaptive margins, and similarity-based matching.


### Model Performance

The trained ResNet-50 + AdaFace model was evaluated using both classification accuracy and face verification metrics.

```text
Total usable images      : 24,497
Total identities/classes : 1,779
Training images          : 19,142
Validation images        : 5,355
````

Final training results:

```text
Training Accuracy        : 99.97%
Validation Accuracy      : 72.94%
Validation Loss          : 9.0715
Mean Gradient Norm       : 27.90
```

Face verification results:

```text
Validation ROC AUC       : 0.9844
Best Similarity Threshold: 0.1907
```

Although the validation classification accuracy was **72.94%** across 1,779 identities, the model achieved a strong **ROC AUC of 0.9844** for face verification. This shows that the learned embedding space is highly effective at separating same-person and different-person face pairs.


## Project 2: Scratch-Built ResNet-18 for Image Classification

This project involves the ground-up implementation of a **ResNet-18** architecture using TensorFlow and Keras to perform high-accuracy binary classification. Unlike standard sequential models that suffer from signal degradation in deeper layers, this project implements **Residual Learning** through skip connections, allowing gradients to flow unimpeded during backpropagation. By building the 18-layer backbone from scratch rather than using pre-trained weights, the project demonstrates how structural innovations like Batch Normalization and Identity Mappings enable the network to converge rapidly and achieve near-perfect metrics on unseen data.



The core of the architecture is the **Residual Block**, which focuses on learning the difference (residual) between the input and output rather than the entire mapping from scratch. Each block consists of two $3 \times 3$ convolutional layers paired with **Batch Normalization** to stabilize training. When spatial dimensions decrease, a $1 \times 1$ convolution is used in the skip connection to match the shortcut's dimensions to the main path. The model concludes with a **Global Average Pooling** layer and a 50% Dropout rate, effectively regularizing the 11.3 million parameters to prevent overfitting.



### 💻 Architecture Snippet
```python
def residual_block(x, filters, stride=1):
    shortcut = x
    # Adjust shortcut if dimensions change
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Main Path
    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    return layers.ReLU()(layers.Add()([x, shortcut]))
```

### 📊 Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **96.34%** |
| **AUC-ROC** | **99.63%** |
| **Precision** | **98.55%** |
| **Recall** | **94.06%** |
| **Test Loss** | **0.0926** |
