# emit-methane-plume-segmentation
Methane plume segmentation in EMIT imagery using CNN–Mamba localization and conditional diffusion refinement.
# Methane Plume Segmentation in EMIT Imagery

This repository accompanies our study of methane plume segmentation using observations from NASA’s Earth Surface Mineral Dust Source Investigation (EMIT).

## Method

The framework delineates methane plumes in two stages. A CNN–Mamba network uses 41 shortwave infrared bands, three RGB channels, and a methane enhancement (ENH) map to estimate a plume probability for each pixel. A conditional diffusion refiner then uses the same imagery and the initial probability map to predict a residual correction.

<img width="1004" height="433" alt="image" src="https://github.com/user-attachments/assets/566b111c-c2d9-4e7b-b174-c03d60506768" />


## Dataset

The EMIT dataset contains 4,372 patches of 512 × 512 pixels. Of these, 2,366 contain at least one labeled plume pixel and 2,006 contain none. All patches from a given EMIT acquisition are assigned to the same training, validation, or test subset.

## Planned release

- Training, inference, and evaluation code
- Model configurations and trained weights
- Patch identifiers and acquisition-level training, validation, and test split lists
- Data access and preprocessing instructions
