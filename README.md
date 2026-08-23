# ir2rgb

Infrared-to-RGB image translation project.
# 🛰️ ir2rgb: Satellite Thermal-IR to RGB Image Synthesis

[![SIH 2026](https://img.shields.io/badge/SIH-2026-orange.svg)](https://sih.gov.in/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Paired Thermal-to-Optical Translation Engine for Landsat Imagery using Conditional Generative Adversarial Networks (Pix2Pix)**  
> *Developed for Smart India Hackathon (SIH)*  
> **Team:** `I.R.I.S. — Infrared Radiance Iterative Spectral-enhancer`

---

## 📌 Executive Summary

Thermal infrared (TIR) satellite imagery captures land surface temperature and heat emission signatures day and night, but lacks the rich semantic visual cues of optical RGB imagery. **`ir2rgb`** is an end-to-end deep learning pipeline built to synthesize high-fidelity 3-band (RGB) optical surface representations directly from single-band satellite thermal measurements.

Developed by team **I.R.I.S.**, the core engine leverages a **Conditional GAN (Pix2Pix)** framework with a U-Net Generator and a PatchGAN Discriminator. The model learns complex non-linear spatial and radiometric dependencies between thermal surface emission patches and visible land surface reflectance.

---

## ✨ Key Features

* **USGS Landsat Calibration Pipeline:** Automated conversion of raw Collection 2 Level-2 Digital Numbers ($\text{DN}$) to absolute Surface Temperature (Kelvin) and Surface Reflectance.
* **Geospatial Resampling & Alignment:** Native handling of GeoTIFF spatial metadata, Coordinate Reference System (CRS) transformations, and patch cropping via `rasterio`.
* **cGAN Architecture (Pix2Pix):** 
  * **Generator:** U-Net encoder-decoder with skip connections to preserve edge geometry.
  * **Discriminator:** $70 \times 70$ PatchGAN conditional architecture for localized texture fidelity.
* **Multi-Loss Optimization:** Objective function combining conditional Binary Cross-Entropy (Adversarial Loss), pixel-wise $L_1$ Reconstruction Loss, and Perceptual/Structural Similarity.
* **Sliding-Window Scene Reconstruction:** Full-scene GeoTIFF inference with overlapping tile processing and Gaussian blending to eliminate boundary stitching artifacts.
* **Interactive Glassmorphic Web Dashboard:** FastAPI backend serving a responsive frontend with a dynamic before/after split slider for thermal-to-RGB visual validation.

---

## 📁 Repository Directory Structure

```text
ir2rgb/
│
├── data/
│   ├── raw/                  # Raw Landsat Collection 2 GeoTIFF scenes (B10, B4, B3, B2)
│   ├── processed/            # Radiometrically scaled & georeferenced NumPy arrays
│   ├── train/                # Paired training patches (input_thermal/ ↔ target_rgb/)
│   ├── val/                  # Geographically isolated validation scenes
│   └── test/                 # Reserved unseen test scenes for final evaluation
│
├── src/
│   ├── data/
│   │   ├── inspect.py        # Metadata inspection, CRS check, & raster summary
│   │   ├── preprocess.py     # USGS radiometric scaling (Kelvin & Reflectance)
│   │   ├── align.py          # Spatial resampling, cropping, & tile generation
│   │   └── dataset.py        # PyTorch Dataset & DataLoader implementations
│   │
│   ├── models/
│   │   ├── generator.py      # U-Net encoder-decoder generator architecture
│   │   ├── discriminator.py  # PatchGAN conditional discriminator architecture
│   │   └── pix2pix.py        # Integrated conditional GAN trainer wrapper
│   │
│   ├── training/
│   │   ├── train.py          # Training loop execution script
│   │   ├── losses.py         # Adversarial, L1, and structural loss modules
│   │   └── config.py         # Hyperparameter and path configurations
│   │
│   ├── evaluation/
│   │   ├── metrics.py        # PSNR and SSIM quantitative calculation utilities
│   │   └── evaluate.py       # Validation set evaluation & visual report generation
│   │
│   └── inference/
│       └── predict.py        # Whole-scene sliding window inference & GeoTIFF stitching
│
├── checkpoints/              # Model weights & checkpoint state files (.pth)
├── outputs/                  # Reconstruction outputs & evaluation visual plots
│
├── app/
│   ├── backend/
│   │   ├── main.py           # FastAPI server routes & REST API handlers
│   │   └── model_service.py  # PyTorch model loading & inference runner
│   │
│   └── frontend/
│       ├── index.html        # Glassmorphic UI with split-image slider
│       ├── style.css         # Responsive styling & glass visual effects
│       └── script.js         # API client & interactive events
│
├── notebooks/                # Jupyter / Google Colab notebooks for training
├── config.yaml               # Pipeline hyperparameters & directory setup
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md                 # Project documentation
