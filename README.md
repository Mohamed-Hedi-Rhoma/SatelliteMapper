# Physics-Informed CycleGAN for Satellite Image Translation

A deep learning architecture for translating between Landsat-8 and Sentinel-2 satellite imagery with physically consistent results.

## Overview

This project implements a physics-informed CycleGAN that translates between Landsat-8 and Sentinel-2 satellite imagery. The model incorporates physical constraints such as spectral response functions, viewing geometry, and multi-resolution handling to create translations that not only look realistic but also respect the physical characteristics of both satellite sensors.

## Features

- **Bidirectional Translation**: Convert Landsat-8 to Sentinel-2 and vice versa
- **Multi-Resolution Handling**: Properly manages different resolutions (30m for Landsat-8, 10m/20m for Sentinel-2)
- **Physics-Informed Components**:
  - Spectral Response Function (RSR) integration
  - Angular information conditioning
  - Hyperspectral intermediate representation
- **Residual Architecture**: Improves gradient flow and feature learning
- **Attention Mechanism**: Captures long-range dependencies in feature maps

## Model Architecture

### Generators

**Landsat-8 to Sentinel-2 Generator**  
Input: Landsat-8 image [B, 6, 128, 128], angular information  
↓  
Initial Convolution + Residual Blocks + Attention  
↓  
Hyperspectral Projection  
↓  
Multi-Resolution Upsampling (3× for VNIR, 1.5× for SWIR)  
↓  
Spectral Response Layer  
↓  
Output: Sentinel-2 bands at native resolutions  
- VNIR: [B, 4, 384, 384] (10m)  
- SWIR: [B, 2, 192, 192] (20m)  

**Sentinel-2 to Landsat-8 Generator**  
Input: Sentinel-2 VNIR [B, 4, 384, 384], SWIR [B, 2, 192, 192], angular information  
↓  
Separate Processing + Resolution Alignment (3× down for VNIR, 1.5× down for SWIR)  
↓  
Feature Fusion + Residual Blocks + Attention  
↓  
Hyperspectral Projection  
↓  
Spectral Response Layer  
↓  
Output: Landsat-8 image [B, 6, 128, 128]  

### Discriminators

- **Sentinel-2 Discriminators**:
  - VNIR Discriminator: For 10m bands (4 channels at 384×384)
  - SWIR Discriminator: For 20m bands (2 channels at 192×192)
- **Landsat-8 Discriminator**:
  - For all 6 bands at 128×128 resolution

All discriminators use the PatchGAN architecture with residual blocks and conditional normalization.

## Installation
```bash
# Install pixi package manager
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies
pixi install


## Data Preparation

1. Download data using:
python get_data_sentinel2.py
python get_data_landsat8.py

(Modify the sites in the code as needed)

2. Use the preprocessing and preparation code to create .pth files for train, validation, and test sets.