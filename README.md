# Learning Unknown Probability Density Function using GAN

## 📌 Objective

The objective of this project is to learn an unknown probability density function (PDF) of a transformed random variable using a Generative Adversarial Network (GAN), without assuming any analytical form of the distribution.

The model learns the distribution purely from data samples.

---

## 📊 Dataset

- Dataset: India Air Quality Dataset (Kaggle)
- Feature Used: NO2 concentration
- Link: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

Only the NO2 column is used for modeling.

---

## 🔄 Step 1: Transformation

Each NO2 value \( x \) is transformed using:

\[
z = x + a_r \sin(b_r x)
\]

where:

\[
a_r = 0.5 (r \mod 7)
\]
\[
b_r = 0.3 ((r \mod 5) + 1)
\]

For my roll number:

- a_r = 3.0
- b_r = 1.2

This produces a transformed variable \( z \) whose distribution is unknown.

---

## 🤖 Step 2: GAN Architecture

The GAN consists of:

### Generator
- Input: Random noise \( \epsilon \sim N(0,1) \)
- 2 Hidden layers (ReLU activation)
- Output: 1-dimensional sample

### Discriminator
- Input: 1D sample (real or fake)
- 2 Hidden layers (LeakyReLU activation)
- Output: Probability (Sigmoid)

### Training Details
- Loss: Binary Cross Entropy
- Optimizer: Adam
- Epochs: 2000
- Latent dimension: 5

The GAN learns the distribution directly from samples of transformed variable \( z \).

---

## 📈 Step 3: PDF Approximation

After training:

1. Large number of samples generated from Generator
2. Kernel Density Estimation (KDE) used to approximate PDF
3. Estimated PDF plotted against real histogram

---

## 📊 Results

- The GAN successfully captures the dominant mode of the distribution.
- Central density region closely matches real data.
- Minor deviation observed in extreme tail values.
- No mode collapse observed.
- Training remained stable throughout epochs.

---

## 🔍 Observations

### Mode Coverage
The model captures the main peak effectively but slightly underestimates rare extreme values.

### Training Stability
Loss oscillations indicate stable adversarial equilibrium without divergence.

### Distribution Quality
Generated samples approximate the empirical distribution well in high-density regions.

---

## 🛠 Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

---

## 📁 Repository Structure

├── GAN_PDF_Estimation_NO2.ipynb
├── README.md
└── data.csv

---

## 🎯 Key Learning

- Learning unknown PDFs using implicit generative modeling
- Understanding adversarial training dynamics
- Practical density estimation using GANs
- Mode coverage and training stability analysis
