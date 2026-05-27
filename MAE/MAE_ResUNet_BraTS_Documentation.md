# 🧠 MAE Pretraining for Medical Image Segmentation — BraTS Edition
## Complete Technical Documentation

---

## Table of Contents
1. [Project Overview](#overview)
2. [Pipeline Summary](#pipeline)
3. [Section-by-Section Code Explanation](#sections)
   - [Section 1: Environment Setup & Imports](#section1)
   - [Section 2: Configuration](#section2)
   - [Section 3: Dataset Loading](#section3)
   - [Section 4: Medical Augmentations](#section4)
   - [Section 5: MAE Pretraining](#section5)
   - [Section 6: ResUNet Fine-Tuning](#section6)
   - [Section 7: Evaluation Metrics](#section7)
   - [Section 8: Results & Visualizations](#section8)
4. [Architecture Descriptions](#architectures)
5. [Key Design Decisions](#design)

---

## 1. Project Overview <a name="overview"></a>

This notebook implements a **Self-Supervised Learning (SSL)** pipeline for brain tumour segmentation on the BraTS dataset. The central idea is to pre-train a Vision Transformer (ViT) using **Masked Autoencoders (MAE)** on *unlabeled* MRI slices, then leverage the learned representations to improve a **ResUNet** segmentation model via knowledge transfer and fine-tuning with limited labeled data.

**Problem addressed:** Medical image annotation is expensive and time-consuming. SSL pre-training allows the model to learn rich visual representations from abundant unlabeled scans, reducing dependence on large labeled datasets.

---

## 2. Pipeline Summary <a name="pipeline"></a>

```
BraTS NIfTI Volumes
        │
        ├─── Unlabeled slices ──► MAE Pre-training (ViT-based)
        │                              │
        │                         Encoder learns rich
        │                         MRI representations
        │                              │
        └─── Labeled pairs ────► Knowledge Transfer
                                  (pseudo-labeling via kNN
                                   in MAE feature space)
                                       │
                                  ResUNet Fine-tuning
                                  (ResNet34 Encoder +
                                   UNet Decoder)
                                       │
                              Evaluation: Dice / IoU / HD95
                                       │
                              Baseline: Plain UNet
                              (trained from scratch)
```

---

## 3. Section-by-Section Code Explanation <a name="sections"></a>

---

### Section 1: Environment Setup & Imports <a name="section1"></a>

**Purpose:** Install dependencies and configure the runtime environment.

**Key packages:**
- `torch`, `torchvision` — deep learning framework
- `timm` — pretrained model zoo (ResNet34 backbone)
- `nibabel` — load NIfTI (.nii.gz) brain scan files
- `monai` — medical imaging transforms (Gibbs noise, bias field, etc.)
- `einops` — tensor reshaping utilities
- `scipy` — Hausdorff distance computation

**Device selection logic:**
```python
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else 'cpu'
)
```
Automatically selects NVIDIA GPU → Apple Silicon → CPU.

**Reproducibility:** `seed_everything(42)` seeds Python, NumPy, and PyTorch random generators.

**Cell timer:** A custom IPython hook (`pre_run_cell` / `post_run_cell`) prints elapsed time after every cell.

---

### Section 2: Configuration <a name="section2"></a>

**Purpose:** Define all hyperparameters as YAML strings parsed into Python dictionaries.

#### MAE Configuration
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `backbone` | vit_base_patch16 | ViT-B/16 architecture |
| `image_size` | 224 | Input resolution |
| `patch_size` | 16 | 14×14 = 196 patches per image |
| `embed_dim` | 768 | Token embedding dimension |
| `encoder_depth` | 12 | Number of transformer encoder layers |
| `encoder_heads` | 12 | Multi-head attention heads |
| `decoder_embed_dim` | 512 | Decoder projection dimension |
| `decoder_depth` | 8 | Decoder transformer layers |
| `mask_ratio` | 0.75 | 75% of patches masked during pre-training |
| `epochs` | 50 | Pre-training epochs (100–400 recommended for full run) |
| `batch_size` | 64 | |
| `learning_rate` | 1.5e-4 | AdamW LR |
| `optimizer` | adamw | With β=(0.9, 0.95) |
| `scheduler` | cosine | With warmup |

#### Fine-Tuning Configuration
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `encoder` | resnet34 | ResNet34 pretrained on ImageNet |
| `num_classes` | 4 | BG, NCR, ED, ET |
| `epochs` | 50 | |
| `batch_size` | 16 | |
| `learning_rate` | 3e-4 | |
| `encoder_lr_scale` | 0.5 | Encoder LR = base_lr × 0.5 |
| `loss` | dice_ce | Combined Dice + Cross-Entropy |
| `label_smoothing` | 0.05 | Prevents overconfidence |
| `use_class_weights` | true | Upweights rare tumour classes |
| `label_fractions` | [0.05, 1.0] | Train at 5% and 100% label budgets |

**`DEMO_EPOCHS = 10`** — quick override for smoke-testing; set to `None` for full training.

---

### Section 3: Dataset Loading — BraTS Only <a name="section3"></a>

**Purpose:** Scan the BraTS directory, build slice-level datasets for pre-training and fine-tuning.

#### Expected BraTS Directory Layout
```
data/
  BraTS20_Training_001/
    BraTS20_Training_001_t1n.nii.gz
    BraTS20_Training_001_t1c.nii.gz
    BraTS20_Training_001_t2w.nii.gz
    BraTS20_Training_001_t2f.nii.gz
    BraTS20_Training_001_seg.nii.gz
  BraTS20_Training_002/ ...
```

#### Key Functions

**`load_nifti_volume(path)`**
- Loads a `.nii.gz` file using `nibabel`, returns a float32 NumPy array of shape `(H, W, D)`.

**`normalize_volume(vol, clip_percentile=(0.5, 99.5))`**
- Clips intensity outliers at the 0.5th and 99.5th percentile then rescales to [0, 1].
- The percentile clipping handles scanner artifacts and bright spots that would otherwise dominate normalization.

**`scan_brats_directory(brats_root, modalities, seg_modality)`**
- Walks all case subdirectories and collects:
  - `unlabeled_paths` — all modality NIfTI files (for MAE pre-training, no masks needed)
  - `labeled_pairs` — `(image_path, mask_path)` tuples (for fine-tuning)
- Raises `RuntimeError` if no data is found (no synthetic fallback).

#### `BraTSUnlabeledDataset`
- Indexes all axial slices across NIfTI volumes.
- Filters out mostly-background slices with `mean < 0.05` (foreground ratio threshold).
- Each `__getitem__` call: loads the volume, extracts slice `z`, resizes to 224×224, returns a `(1, 224, 224)` tensor.

#### `BraTSLabeledDataset`
- Indexes `(case_index, slice_z)` pairs.
- Label remapping: BraTS 2020 uses label `4` for Enhancing Tumor (ET), remapped to `3` for 4-class indexing.
- Augmentations applied inline: flips, 90° rotations, Gaussian noise, intensity shift, gamma correction, random erasing.

#### `build_datasets()`
- Calls `scan_brats_directory`, constructs both datasets.
- Splits labeled data 80/20 train/val using `torch.Generator` for reproducibility.

---

### Section 4: Medical-Specific Augmentations <a name="section4"></a>

**Purpose:** Augment unlabeled slices specifically for MAE pre-training in the brain MRI domain.

Each augmentation is an `nn.Module` composable in `nn.Sequential`:

| Class | Effect | Medical Relevance |
|-------|--------|-------------------|
| `GaussianNoise(std=0.05)` | Adds random Gaussian noise | Simulates scanner noise |
| `RandomGammaCorrection(0.7–1.5)` | Non-linear intensity transform | MRI scanner variability |
| `RandomIntensityShift(shift=0.1)` | Linear brightness/contrast | Between-scanner differences |
| `RandomElasticDeformation(α=50, σ=5)` | Smooth spatial warp | Anatomical variability |
| `RandomGhosting(intensity=0.3, n=3)` | Ghosting artifact via rolling | MRI motion artifacts |
| `RandomHorizontalFlip(p=0.5)` | Mirror flip | Bilateral brain symmetry |
| `RandomVerticalFlip(p=0.5)` | Vertical flip | Slice orientation variation |

**`get_mae_augmentation()`** assembles a subset of these into `nn.Sequential` for efficient batched application.

**`AugmentedDataset`** wraps any `Dataset` and applies the augmentation pipeline in `__getitem__`.

---

### Section 5: MAE Pretraining <a name="section5"></a>

**Purpose:** Implement and train a Masked Autoencoder on unlabeled BraTS slices.

#### Component: `PatchEmbed`
```
Input:  (B, 1, 224, 224)
Conv2d(1, 768, 16, stride=16)  →  (B, 768, 14, 14)
Flatten + Transpose            →  (B, 196, 768)
Output: (B, 196, 768)  — 196 patch tokens, each of dim 768
```

#### Component: `get_2d_sincos_pos_embed(embed_dim, grid_size)`
- Generates fixed (non-learnable) 2D sinusoidal positional embeddings.
- Encodes row and column positions separately using different frequency sinusoids.
- Concatenates row and column embeddings → shape `(grid_size², embed_dim)`.

#### Component: `MAEEncoder`

**Architecture:**
```
Input image (B, 1, 224, 224)
   ↓ PatchEmbed
Patch tokens (B, 196, 768) + sincos pos embed
   ↓ Random masking (75%)
Visible tokens (B, 49, 768) + CLS token
   ↓ 12-layer Transformer (d=768, h=12, FFN=3072)
   ↓ LayerNorm
Latent (B, 50, 768) — 49 visible + 1 CLS
```

**Masking algorithm:**
1. Generate random noise per token: `noise = torch.rand(B, N)`
2. Sort tokens by noise value
3. Keep the top `N × (1 − mask_ratio)` = 49 tokens (25%)
4. Store `ids_restore` (argsort of sort indices) to reconstruct original ordering in decoder
5. `mask` tensor: `True` = masked position

#### Component: `MAEDecoder`

**Architecture:**
```
Encoder latent (B, 50, 768)
   ↓ Linear projection (768 → 512)
Decoder tokens + learnable mask tokens
   ↓ Restore to original patch order using ids_restore
   ↓ Add decoder sincos positional embedding
   ↓ 8-layer Transformer (d=512, h=16, FFN=2048)
   ↓ LayerNorm
   ↓ Linear (512 → 16×16×1 = 256)
Predictions (B, 196, 256)
```

**Mask tokens:** Learnable parameter `(1, 1, 512)` broadcast to fill all masked positions. The decoder must learn to predict the original content at each masked location.

#### Component: `MaskedAutoEncoder.forward()`
1. Encodes with masking → `latent, mask, ids_restore`
2. Decodes → `pred` of shape `(B, 196, 256)` (each token predicts its 16×16 pixel patch)
3. Patchifies the original image → `target (B, 196, 256)`
4. **Normalizes target patches:** `(target − mean) / std` — patch normalization stabilizes training
5. Computes MSE only on masked tokens: `loss = (mse * mask).sum() / mask.sum()`

#### Training Loop `pretrain_mae()`
- **Optimizer:** AdamW with β=(0.9, 0.95), lr=1.5e-4, weight_decay=0.05
- **Scheduler:** Cosine decay with linear warmup (`cosine_schedule_with_warmup`)
- **Gradient clipping:** Max norm 1.0
- **Saves:** `mae_pretrained.pth` (model state + config)

#### MAE → ResUNet Knowledge Transfer
Since MAE uses a ViT encoder and ResUNet uses ResNet34, direct weight transfer is impossible. The notebook implements **feature-space pseudo-labeling**:

1. **Feature extraction:** Run MAE encoder at `mask_ratio=0.0` (no masking) → extract CLS token `(N, 768)`
2. **Cosine similarity:** Compute similarity between unlabeled and labeled slice features
3. **Confidence gate:** Keep unlabeled slices whose average similarity to top-k labeled neighbors ≥ 0.70
4. **Semi-supervised expansion:** Add confident pseudo-labeled slices to the training pool (capped at 300)

---

### Section 6: ResUNet Fine-Tuning <a name="section6"></a>

#### Architecture: `ResUNet`

**Encoder (ResNet34 from timm):**
```
Input (B, 1, 224, 224)
   ↓ Conv2d(1→64, 7×7, stride=2) + BN + ReLU   → s0: (B, 64, 112, 112)
   ↓ MaxPool(2)
   ↓ Layer1 (3×BasicBlock, 64→64)               → s1: (B, 64, 56, 56)
   ↓ Layer2 (4×BasicBlock, 64→128, stride=2)    → s2: (B, 128, 28, 28)
   ↓ Layer3 (6×BasicBlock, 128→256, stride=2)   → s3: (B, 256, 14, 14)
   ↓ Layer4 (3×BasicBlock, 256→512, stride=2)   → s4: (B, 512, 7, 7)
```

**1-channel adaptation:** The 3-channel ImageNet conv weights are averaged across the channel axis and divided by `in_chans` to initialize the single-channel conv without losing scale.

**Decoder (`ResUNetDecodeBlock`):**
Each block:
1. `ConvTranspose2d` upsamples and halves channels
2. Spatial size alignment with skip (for odd-dimension safety)
3. Concatenate skip connection
4. Two `ConvBnRelu` (Conv2d → BN → ReLU) layers to fuse features

```
s4 (B, 512, 7, 7)
   ↓ dec4: upsample + concat s3 → (B, 256, 14, 14)
   ↓ dec3: upsample + concat s2 → (B, 128, 28, 28)
   ↓ dec2: upsample + concat s1 → (B, 64, 56, 56)
   ↓ dec1: upsample + concat s0 → (B, 32, 112, 112)
   ↓ final_up: ConvTranspose2d  → (B, 16, 224, 224)
   ↓ final_conv: ConvBnRelu     → (B, 16, 224, 224)
   ↓ head: Conv2d(16→4, 1×1)   → (B, 4, 224, 224)
```

#### Loss Function: `DiceCELoss`
Combined Dice and Cross-Entropy loss:
```
L = 0.5 × DiceLoss + 0.5 × CrossEntropyLoss
```

**`DiceLoss`:** Soft multi-class Dice with optional label smoothing:
```
Dice = 1 − mean_c[(2 × Σ(p_c × t_c) + smooth) / (Σp_c + Σt_c + smooth)]
```

**Class weighting** (`estimate_class_weights()`): Samples up to 500 random masks, computes per-class voxel counts, inverts them, and normalizes so the mean weight = 1.0. This upweights rare tumour classes (NCR, ED, ET).

#### Training Details
- **Differential LR:** Encoder at `base_lr × 0.5`, decoder at `base_lr` (allows encoder to adapt without destroying ImageNet features)
- **OneCycleLR:** 30% linear warmup → cosine decay; much faster convergence than fixed LR
- **AMP (Mixed Precision):** On CUDA, forward/backward at fp16; `GradScaler` manages numeric stability
- **Gradient clipping:** Max norm 1.0
- **Best-model checkpoint:** Tracks `val_dice`, saves best weights at end of training

---

### Section 7: Evaluation Metrics <a name="section7"></a>

**`compute_dice(pred, target, num_classes)`**
- Computes per-class binary Dice, averages across classes
- Formula: `2|P∩T| / (|P| + |T|)`; range [0, 1], higher is better

**`compute_iou(pred, target, num_classes)`**
- Mean IoU (Jaccard index) across classes
- Formula: `|P∩T| / |P∪T|`; range [0, 1], higher is better

**`compute_hausdorff_95(pred_np, target_np)`**
- Surface-based metric: maximum of directed Hausdorff distances between boundary point sets
- Uses binary erosion to extract boundary pixels
- Measures worst-case boundary error in pixels; lower is better
- Applied specifically to the tumour core (class 1) for clinical relevance

**`evaluate_model(model, dataset, num_classes)`**
- Runs inference in batches, aggregates Dice, IoU, and HD95
- Returns a `{dice, iou, hausdorff95}` dictionary

---

### Section 8: Results & Visualizations <a name="section8"></a>

Five plots are generated and saved as PNG files:

1. **`mae_pretrain_loss.png`** — MAE reconstruction loss (MSE on masked patches) vs epoch, with ±5% shaded band
2. **`resunet_vs_unet_curves.png`** — Side-by-side: training loss curves (left) and validation Dice curves (right) for both models across all label fractions
3. **`resunet_vs_unet_bars.png`** — Grouped bar chart comparing final Dice scores at each label fraction
4. **`resunet_vs_unet_predictions.png`** — 4×4 grid: Input MRI / Ground Truth / ResUNet prediction / UNet prediction for 4 validation slices, with color-coded class legend
5. **`mae_reconstructions.png`** — 3×4 grid: Original / Masked (75%) / Reconstructed for 4 BraTS slices

---

## 4. Architecture Descriptions <a name="architectures"></a>

### MAE (Masked Autoencoder)

| Component | Layers | Parameters (full config) |
|-----------|--------|--------------------------|
| PatchEmbed | Conv2d(1,768,16,16) | ~150K |
| Encoder Transformer | 12 × TransformerEncoderLayer | ~85M |
| Decoder Embed | Linear(768→512) | ~393K |
| Decoder Transformer | 8 × TransformerEncoderLayer | ~25M |
| Prediction head | Linear(512→256) | ~131K |
| **Total** | | ~111M |

*Note: Demo configuration uses embed_dim=128, depth=2 for speed.*

### ResUNet (ResNet34 Encoder + U-Net Decoder)

| Stage | Type | Output Shape |
|-------|------|-------------|
| Input | — | (B, 1, 224, 224) |
| Stem | Conv+BN+ReLU | (B, 64, 112, 112) |
| Layer1 | 3× BasicBlock | (B, 64, 56, 56) |
| Layer2 | 4× BasicBlock | (B, 128, 28, 28) |
| Layer3 | 6× BasicBlock | (B, 256, 14, 14) |
| Layer4 (Bottleneck) | 3× BasicBlock | (B, 512, 7, 7) |
| Dec4 | TransConv+Skip+Conv | (B, 256, 14, 14) |
| Dec3 | TransConv+Skip+Conv | (B, 128, 28, 28) |
| Dec2 | TransConv+Skip+Conv | (B, 64, 56, 56) |
| Dec1 | TransConv+Skip+Conv | (B, 32, 112, 112) |
| Final Up | TransConv+BN+ReLU | (B, 16, 224, 224) |
| Head | Conv1×1 | (B, 4, 224, 224) |

Total parameters: ~21.8M (ResNet34 backbone: ~21.3M)

### Baseline U-Net

| Stage | Channels | Output Shape |
|-------|----------|-------------|
| Input | 1 | (B, 1, 224, 224) |
| Enc1 | 1→32 | (B, 32, 224, 224) |
| Enc2 | 32→64 | (B, 64, 112, 112) |
| Enc3 | 64→128 | (B, 128, 56, 56) |
| Enc4 | 128→256 | (B, 256, 28, 28) |
| Bottleneck | 256→512 | (B, 512, 14, 14) |
| Dec4 | 512→256 | (B, 256, 28, 28) |
| Dec3 | 256→128 | (B, 128, 56, 56) |
| Dec2 | 128→64 | (B, 64, 112, 112) |
| Dec1 | 64→32 | (B, 32, 224, 224) |
| Head | 32→4 | (B, 4, 224, 224) |

Total parameters: ~7.7M (trained from scratch)

---

## 5. Key Design Decisions <a name="design"></a>

### Why MAE for Medical Images?
- 75% masking ratio forces the model to learn global context, not just local texture — critical for understanding tumor shape and extent
- No labels required during pre-training, enabling use of all available MRI data
- Patch normalization in the MSE target prevents trivial solutions based on mean intensity

### Why ResNet34 for Fine-Tuning Encoder?
- ImageNet pretraining provides strong low-level feature detectors (edges, textures) that transfer well to MRI
- Substantially fewer parameters than ViT-B while still deep enough for multi-scale feature extraction
- `features_only=True` mode in timm provides clean multi-scale outputs for U-Net skip connections

### Why Not Direct MAE → ResUNet Weight Transfer?
- ViT operates on flat token sequences; ResNet operates on spatially-structured feature maps
- The architectures are fundamentally incompatible for weight-sharing
- The pseudo-labeling approach is the principled way to bridge SSL pretraining to a different fine-tuning backbone

### Class Weighting Strategy
- BraTS is heavily imbalanced: background dominates (~90% of voxels)
- Without weighting, the model learns to predict background and achieves high accuracy but misses tumours
- Inverse-frequency weights ensure rare classes (NCR, ED, ET) contribute equally to the gradient

### OneCycleLR vs CosineAnnealing
- OneCycleLR with 30% warmup trains faster — reaches good solutions in fewer epochs
- Especially important in semi-supervised settings where the training set may be small (5% label fraction)
