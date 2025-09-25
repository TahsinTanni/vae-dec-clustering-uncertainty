# VAE+DEC for Image Clustering with Uncertainty Analysis

This repository contains the implementation and analysis of a **Variational Autoencoder (VAE)** combined with **Deep Embedded Clustering (DEC)** for **unsupervised image clustering**.  
The project also explores **uncertainty quantification** (via entropy and latent variance) to evaluate clustering correctness and model confidence.  

The work was conducted as part of a research/assignment project on **non-deterministic unsupervised learning**.

---

## Project Overview

- **Objective:**  
  Design and implement a **non-deterministic deep clustering model** that leverages VAE’s probabilistic latent space and DEC’s clustering optimization.  
  Compare with a **deterministic baseline (VAE + K-Means)** and analyze whether uncertainty measures can predict clustering errors.

- **Dataset:**  
  [COIL-20 (Columbia Object Image Library)](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)  
  - 20 objects, each with 72 views (total = 1440 images).  
  - Preprocessing: resized to 32×32, converted to grayscale, normalized.  

- **Model Highlights:**  
  - **Encoder:** 3 convolutional layers → latent space (dim=10, mean + log variance).  
  - **Decoder:** linear + transposed convolutions, outputs reconstructed 32×32 grayscale image.  
  - **Clustering layer:** soft assignments with Student’s t-distribution, refined using KL divergence to a target distribution.  
  - **Baseline:** Deterministic VAE (no sampling) + K-Means clustering on latent codes.

- **Loss Function:**  
  \[
  L_{total} = L_{recon} + \beta L_{KL} + \gamma L_{cluster}
  \]
  - Reconstruction loss: Binary Cross-Entropy (BCE)  
  - KL divergence: regularizes latent space  
  - Clustering loss: KL divergence between soft assignments (Q) and target distribution (P)

---

## 📊 Results

- **Performance (VAE+DEC vs Deterministic VAE + K-Means):**

| Metric       | VAE+DEC (Mean ± Std) | Deterministic VAE (Mean ± Std) |
|--------------|----------------------|--------------------------------|
| Accuracy     | **0.85 ± 0.02**      | 0.75 ± 0.03                    |
| NMI          | **0.88 ± 0.015**     | 0.78 ± 0.025                   |
| ARI          | **0.82 ± 0.018**     | 0.72 ± 0.028                   |
| Silhouette   | **0.45 ± 0.01**      | 0.35 ± 0.015                   |

✅ VAE+DEC consistently outperforms deterministic baseline by ~10–15% across all metrics.  
✅ Welch’s t-tests confirm differences are statistically significant (p < 0.05).  

- **Uncertainty Analysis:**  
  - Correlation between **cluster assignment entropy** and correctness: **−0.65**  
  - Correlation between **latent variance** and correctness: **−0.55**  
  - High uncertainty samples = more likely misclustered.  
  - Conclusion: **uncertainty reliably predicts clustering errors**.  

- **Visualizations:**  
  - **t-SNE plots**: VAE+DEC latent space shows cleaner cluster separation than deterministic baseline.  
  - **Loss curves**: VAE+DEC converges stably with joint optimization of reconstruction, KL, and clustering.  
  - **Uncertainty distributions**: right-skewed, with incorrect predictions concentrated in high-uncertainty regions.  

---

## 📂 Repository Structure


```
vae-dec-clustering-uncertainty/
├── vae_dec_clustering.py          # Main PyTorch implementation (optional export from notebook)
├── 22101744_CSE425_project.ipynb  # Jupyter notebook with code & experiments
├── vae_dec_clustering_report.pdf  # Detailed project report
├── README.md                      # Documentation (this file)
├── LICENSE                        # MIT license
└── .gitignore                     # Ignore unnecessary files

---
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- torchvision
- scikit-learn 1.3+
- numpy
- scipy
- matplotlib

**Install dependencies:**

```bash
pip install torch torchvision scikit-learn numpy scipy matplotlib

```


## Usage

**Clone the repository:**

```bash
git clone https://github.com/your-username/vae-dec-clustering-uncertainty.git
cd vae-dec-clustering-uncertainty

```

## Applications

- Automated object categorization (e.g., in computer vision pipelines)  
- Anomaly detection via uncertainty estimation  
- Active learning: prioritize samples with high uncertainty for labeling  
- Data exploration: better understanding of latent space representations  

---

## Future Work

- Explore alternative encoder/decoder architectures  
- Investigate new clustering loss functions & adaptive weighting schemes  
- Apply model to larger & more diverse datasets (images, text, multimodal)  
- Extend to semi-supervised learning with limited labels  
- Develop improved uncertainty utilization for real-world tasks  

