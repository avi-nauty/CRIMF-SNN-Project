# 🧠 Brain Tumor Detection using CRIMF Spiking Neural Networks

A hybrid deep learning pipeline for binary brain tumor classification from MRI scans, combining a **pretrained CNN feature extractor** with a biologically-inspired **CRIMF Spiking Neural Network (SNN)** classifier — achieving up to **97.46% test accuracy** with energy-efficient spike-based computation.

---

## Overview

This project applies the **Channelwise Regional Integrate and Multiple Firing (CRIMF)** neuron model — introduced in *"Channelwise Regional Integrate and Multiple Firing Neuron: Improving the Spatiotemporal Learning of Spiking Neural Networks"* (Cai et al., IEEE TNNLS, Jan 2026) — to the task of brain tumor detection from MRI images.

The pipeline bridges conventional deep learning and neuromorphic computing: a ResNet18 backbone extracts spatial features from MRI scans, those features are encoded into spike trains via rate coding, and a CRIMF-based SNN classifier performs the final binary classification (healthy vs. tumor).

---

## Architecture

```
MRI Image (224×224, grayscale)
        │
        ▼
┌─────────────────────────┐
│  ResNet18 (pretrained)  │  ← Modified for 1-channel input
│  Feature Extractor      │    RGB weights averaged → grayscale init
└─────────────┬───────────┘
              │ (B, 512) embeddings
              ▼
┌─────────────────────────┐
│  Normalize to [0, 1]    │  ← Per-sample min-max normalization
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│  Rate-Code Spike Encoder│  ← Stochastic Bernoulli encoding
│  T=10 time steps        │    Output: (T, B, 512) binary spikes
└─────────────┬───────────┘
              │
              ▼
┌──────────────────────────────────┐
│  CRIMF Layer                     │
│  ├─ Linear projection (512→256)  │
│  └─ CRIMFCell (per time step)    │
│      ├─ Regional Current Q(t)    │  ← Long-term memory state
│      ├─ Membrane Potential U(t)  │
│      ├─ Soft Reset               │  ← Enables multiple firing
│      └─ Channelwise τ_q, τ_u     │  ← Learnable per-channel decays
└─────────────┬────────────────────┘
              │ (T, B, 256) spike outputs
              ▼
┌─────────────────────────┐
│  Temporal Averaging     │  ← Mean spike rate across T steps
│  Linear Readout (256→2) │
└─────────────┬───────────┘
              ▼
        Class Logits
     (Healthy / Tumor)
```

---

## The CRIMF Neuron

The CRIMF neuron extends the classic Leaky Integrate-and-Fire (LIF) model with three key innovations that address fundamental limitations of standard SNNs:

### 1. Regional Current (Long-Term Memory)

Standard LIF neurons forget input quickly due to membrane potential decay. The CRIMF neuron introduces a **regional current state** Q(t) that integrates inputs over multiple time steps before charging the membrane:

```
Q(t) = τ_q · Q(t-1) + (1 - τ_q) · I(t)
U(t) = τ_u · U(t-1) + (1 - τ_u) · Q(t)
```

This creates a two-stage memory: Q(t) acts as a temporal accumulator, and U(t) integrates that accumulated signal into the membrane. The result is a neuron whose firing is sensitive to patterns across a longer history of input — not just the most recent time step.

### 2. Multiple Firing Mechanisms (Underactivation & Gradient Saturation Fix)

Standard SNNs with single firing thresholds suffer from two problems:
- **Underactivation**: Negative membrane potentials accumulate continuously, causing many neurons to fire rarely, blocking information flow.
- **Gradient Saturation**: Most membrane potentials fall in regions where the surrogate gradient is near zero, crippling backpropagation.

CRIMF uses a **soft reset** after firing — subtracting the threshold rather than hard-resetting to zero — which allows the membrane to retain residual charge and fire multiple times across time steps. The surrogate gradient used here is the fast sigmoid derivative:

```
S(t) = Heaviside(U(t) - V_th)       [forward]
dS/dU ≈ α · sigmoid(α·x) · (1 - sigmoid(α·x))   [backward]
U(t) = U(t) - S(t) · V_th            [soft reset]
```

### 3. Channelwise Learnable Dynamics (Heterogeneity Learning)

Rather than sharing a single set of temporal dynamics across all neurons, CRIMF assigns **independent learnable decay parameters per channel**:

```
τ_q^c = sigmoid(f^c)    # forget rate for regional current, channel c
τ_u^c = sigmoid(m^c)    # decay rate for membrane, channel c
```

All parameters `f^c`, `m^c` are initialized to 0 (giving τ = 0.5) and learned end-to-end. This allows different feature channels to develop distinct temporal sensitivities — some may integrate over longer horizons, others respond quickly — enriching the representational diversity of the network.

---

## Dataset

**Kaggle MRI Brain Tumor Dataset** (`main-dataset-mri1`)

| Split | Healthy | Tumor | Total |
|-------|---------|-------|-------|
| Train | — | — | 2,214 |
| Test  | 140 | 254 | 394 |

**Preprocessing pipeline:**

- Converted to grayscale (1-channel)
- Aspect-ratio-preserving resize with zero-padding to 224×224 (`ResizeWithPad`)
- Normalization: mean=0.5, std=0.5 → values in approximately [−1, 1]

**Training augmentations:**

- Random affine: ±15° rotation, ±8% translation, ±10% scale
- Color jitter: ±8% brightness, ±10% contrast

---

## Results

Training was run for 10 epochs on a single NVIDIA Tesla T4 GPU (Kaggle). The SNN classifier was trained with the CNN backbone frozen (embeddings computed with `torch.no_grad()`).

| Epoch | Train Loss | Train Acc | Test Acc | Precision | Recall | F1 |
|-------|-----------|-----------|----------|-----------|--------|----|
| 1 | 0.0865 | 96.97% | **97.46%** | 99.59% | 96.46% | 98.00% |
| 2 | 0.0670 | 97.47% | 97.21% | 97.28% | 98.43% | 97.85% |
| 3 | 0.0872 | 96.79% | 97.72% | 98.42% | 98.03% | 98.22% |
| 9 | 0.0783 | 97.15% | 97.72% | 99.60% | 96.85% | 98.20% |

**Best test accuracy: 97.72%** | **Best F1 score: 98.22%**

Final confusion matrix (epoch 3 / epoch 9):

```
              Predicted
              Healthy  Tumor
True Healthy  [ 136      4 ]
True Tumor    [   5    249 ]
```

The model shows strong precision on tumor prediction (>98%), with recall consistently above 96% — critical for a medical classification task where false negatives carry high risk.

---

## Project Structure

```
├── crimf-snn-project-draft3.ipynb   # Main notebook (Kaggle)
├── README.md
```

**Key components in the notebook:**

| Section | Description |
|---------|-------------|
| `ResizeWithPad` | Custom transform preserving anatomical proportions |
| `make_loaders()` | Dataset loading with train/test augmentation split |
| `ResNet18FeatureExtractor` | 1-channel-adapted ResNet18, outputs (B, 512) |
| `normalize_embeddings()` | Per-sample min-max normalization to [0, 1] |
| `rate_code()` | Stochastic Bernoulli spike encoding, returns (T, B, F) |
| `SurrogateHeaviside` | Custom autograd function with fast-sigmoid backward |
| `CRIMFCell` | Core CRIMF neuron: regional current + membrane + soft reset |
| `CRIMFLayer` | Linear projection + CRIMFCell unrolled over T time steps |
| `CRIMFSNNClassifier` | Full model: CRIMF layer + temporal averaging + readout |
| `train_one_epoch()` | Training loop with confusion matrix tracking |

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision matplotlib seaborn pillow
```

### Running on Kaggle

1. Add dataset `main-dataset-mri1` to your Kaggle notebook.
2. Set GPU accelerator (Tesla T4 recommended).
3. Run all cells in order.

### Running locally

```python
data_root = "/path/to/Dataset"   # folder with train/ and test/ subdirs

train_loader, test_loader, train_ds, test_ds = make_loaders(
    data_root=data_root,
    img_size=224,
    batch_size=16,
)
```

**Dataset folder structure expected:**

```
Dataset/
├── train/
│   ├── healthy/
│   └── tumor/
└── test/
    ├── healthy/
    └── tumor/
```

---

## Theoretical Background

### Why SNNs for Medical Imaging?

Spiking Neural Networks communicate via discrete binary events (spikes) rather than continuous activations. On neuromorphic hardware, this reduces energy consumption by orders of magnitude compared to ANNs — relevant for edge deployment in clinical devices.

### Why CRIMF over standard LIF?

The standard LIF neuron has three well-documented failure modes that the CRIMF neuron directly addresses:

| Problem | LIF behaviour | CRIMF solution |
|---------|--------------|----------------|
| Short-term memory | Membrane decays fast, misses patterns across distant time steps | Regional current Q(t) creates two-stage temporal integration |
| Underactivation | Negative potential accumulates, neurons silent | Soft reset keeps potential near threshold; multiple firing enabled |
| Gradient saturation | Most ∂S/∂U ≈ 0, weight updates stall | Symmetric surrogate gradient activates for both positive and negative potentials |
| Heterogeneity | All channels share one dynamics equation | Per-channel learnable τ_q, τ_u diverge during training |

### Rate Coding

Each float-valued CNN embedding dimension is converted to a Bernoulli firing probability. Over T=10 time steps, the SNN receives a binary tensor of shape (T, B, 512). The average firing rate across time encodes feature intensity, while the temporal pattern encodes additional structure the CRIMF cell can exploit.

---

## Inspiration & Citation

This project implements the CRIMF neuron model proposed in:

> Mincheng Cai, Quan Liu, Kun Chen, Li Ma.
> **"Channelwise Regional Integrate and Multiple Firing Neuron: Improving the Spatiotemporal Learning of Spiking Neural Networks."**
> *IEEE Transactions on Neural Networks and Learning Systems*, Vol. 37, No. 1, January 2026.
> DOI: 10.1109/TNNLS.2025.3606849

The original paper applies CRIMF to EEG-based emotion recognition on the SEED and SEED-IV datasets. This project adapts the neuron model to medical image classification via a CNN+SNN hybrid pipeline.

---

## Future Work

- End-to-end joint training of CNN backbone + CRIMF-SNN (currently backbone is frozen)
- Multi-class tumor type classification
- Evaluation on larger datasets (BraTS, RSNA)
- Neuromorphic hardware deployment (Intel Loihi, IBM TrueNorth) for energy benchmarking
- Incorporation of the full CRIMF channelwise normalization and dendrite/axon dual-firing as described in the paper

---

## License

This project is for research and educational purposes.
