# 🖼️ Image Denoising with Autoencoder (PyTorch)

This project demonstrates how to build and train a simple **Autoencoder** using **PyTorch** to remove Gaussian noise from handwritten digit images in the **MNIST** dataset. The autoencoder learns to reconstruct clean images from noisy inputs. :contentReference[oaicite:0]{index=0}

---

# 📌 Overview

An **Autoencoder** is an unsupervised neural network that learns a compressed representation of input data and reconstructs it as accurately as possible.

In this project:

- Clean MNIST images are used as target outputs.
- Gaussian noise is added to the input images.
- The autoencoder learns to remove the noise.
- Mean Squared Error (MSE) is used as the loss function.

---

# 📂 Dataset

- **Dataset:** MNIST
- **Training Images:** 60,000
- **Image Size:** 28 × 28 pixels
- **Channels:** 1 (Grayscale)

---

# 📚 Libraries Used

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

---

# 🔄 Data Preprocessing

Two versions of the MNIST dataset are loaded.

## 1. Noisy Images

The noisy dataset is created by:

- Converting images to tensors.
- Adding Gaussian noise.
- Clamping pixel values between 0 and 1.

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.5 * torch.randn_like(x)),
    transforms.Lambda(lambda x: torch.clamp(x, 0., 1.))
])
```

### Why clamp?

Adding noise may produce pixel values outside the valid range.

Example:

```
-0.3 → 0
1.4  → 1
```

This keeps all pixel values valid.

---

## 2. Clean Images

The clean dataset only converts images to tensors.

```python
transforms.ToTensor()
```

These images act as the target outputs during training.

---

# 📦 DataLoader

```python
DataLoader(..., batch_size=128)
```

Each training batch contains:

- 128 noisy images
- 128 corresponding clean images

The two dataloaders are combined using:

```python
zip(noisy_image, clean_image)
```

This ensures every noisy image is paired with its clean version.

---

# 🧠 Model Architecture

```
Input Image
      │
      ▼
Flatten
      │
      ▼
Linear (784 → 128)
      │
      ▼
ReLU
      │
      ▼
Linear (128 → 64)
      │
Compressed Representation
      ▼
Linear (64 → 128)
      │
      ▼
ReLU
      │
      ▼
Linear (128 → 784)
      │
      ▼
Sigmoid
      │
      ▼
Reshape
      │
      ▼
Output Image (1 × 28 × 28)
```

---

# 🔹 Encoder

The encoder compresses the image into a lower-dimensional representation.

```python
Flatten
Linear(784 → 128)
ReLU
Linear(128 → 64)
ReLU
```

Output:

```
784 pixels

↓

64 features
```

---

# 🔹 Decoder

The decoder reconstructs the original image.

```python
Linear(64 → 128)
ReLU
Linear(128 → 784)
Sigmoid
```

The final output is reshaped back into:

```
(1,28,28)
```

---

# 🚀 Forward Pass

```python
def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x.view(-1,1,28,28)
```

Flow:

```
Noisy Image

↓

Encoder

↓

Latent Vector

↓

Decoder

↓

Clean Image
```

---

# 📉 Loss Function

```python
criterion = nn.MSELoss()
```

Mean Squared Error compares:

```
Predicted Clean Image

vs

Actual Clean Image
```

Lower MSE indicates better reconstruction quality.

---

# ⚡ Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam updates the network weights to minimize reconstruction error.

---

# 🔁 Training Loop

For each epoch:

1. Load a batch of noisy images.
2. Load the matching clean images.
3. Pass noisy images through the autoencoder.
4. Compute reconstruction loss.
5. Clear previous gradients.
6. Compute gradients.
7. Update weights.

```python
outputs = model(noisy_imgs)

loss = criterion(outputs, clean_imgs)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

---

# 📊 Training Output

After each epoch:

```python
print(f"loss is {loss} in epoch {epoch+1}")
```

Example:

```
loss is 0.0182 in epoch 1

loss is 0.0104 in epoch 2

loss is 0.0071 in epoch 3
```

A decreasing loss indicates the model is learning to reconstruct cleaner images.

---

# 📝 Key Concepts Learned

- Autoencoder architecture
- Image denoising
- Gaussian noise augmentation
- Encoder and decoder networks
- Flattening images
- Latent feature representation
- Image reconstruction
- Mean Squared Error (MSE)
- Adam optimizer
- Forward propagation
- Backpropagation
- MNIST dataset
- DataLoader
- PyTorch Sequential model

---

# ▶️ Run the Project

```bash
python Image_Denoising_with_Autoenders.py
```

---

# 🎯 Expected Outcome

After training, the autoencoder learns to:

- Accept noisy handwritten digit images.
- Extract meaningful features.
- Remove Gaussian noise.
- Reconstruct cleaner versions of the original images.

As training progresses, the reconstruction quality improves while the MSE loss decreases.

---

# 📖 Conclusion

This project provides a simple introduction to image denoising using autoencoders in PyTorch. By training on noisy and clean MNIST image pairs, the model learns to reconstruct cleaner images, demonstrating how neural networks can be used for image restoration and feature learning.
