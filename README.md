# emit-methane-plume-segmentation
Methane plume segmentation in EMIT imagery using CNN–Mamba localization and conditional diffusion refinement.
# Methane Plume Segmentation in EMIT Imagery

This repository accompanies our study of methane plume segmentation using observations from NASA’s Earth Surface Mineral Dust Source Investigation (EMIT).

## Method

The framework delineates methane plumes in two stages. A CNN–Mamba network uses 41 shortwave infrared bands, three RGB channels, and a methane enhancement (ENH) map to estimate a plume probability for each pixel. A conditional diffusion refiner then uses the same imagery and the initial probability map to predict a residual correction.

<img width="1004" height="433" alt="image" src="https://github.com/user-attachments/assets/566b111c-c2d9-4e7b-b174-c03d60506768" />
Figure 1. Two-stage framework for methane plume segmentation. The CNN–Mamba network maps aligned SWIR, RGB, and ENH inputs to an initial plume probability map P. The conditional diffusion refiner uses the imagery and P to predict the residual added to P.

<img width="1004" height="483" alt="image" src="https://github.com/user-attachments/assets/c8d2b0bd-8b8b-4a6e-bba6-dabca64dced0" />
Figure 2. Architecture of the Stage-1 CNN-Mamba discriminative segmentation network. The encoder-decoder framework integrates a Mamba global-context module at the bottleneck to enhance long-range dependency modeling and outputs the Stage-1 plume prediction.

## Dataset

The EMIT dataset contains 4,372 patches of 512 × 512 pixels. Of these, 2,366 contain at least one labeled plume pixel and 2,006 contain none. All patches from a given EMIT acquisition are assigned to the same training, validation, or test subset.
<img width="1004" height="670" alt="image" src="https://github.com/user-attachments/assets/e6ca3d55-14c9-420b-8721-ad029536aed0" />
Figure 3. Global distribution of the collected EMIT methane plume samples.

## Data availability

Because of its size, the processed EMIT patch dataset is distributed directly by the corresponding author. To obtain the 45-channel image patches and their corresponding reference masks, please email [quyu0813@m.scnu.edu.cn](mailto:quyu0813@m.scnu.edu.cn).


## Planned release

- Training, inference, and evaluation code
- <img width="589" height="783" alt="image" src="https://github.com/user-attachments/assets/5a01afc4-c453-484a-b144-942d94fb8448" />
Figure 4. Construction of patch reference masks for four EMIT acquisitions. Each row shows the same spatial window across three panels. The left column shows methane enhancement (ENH) on a common color scale, with missing values shown in white. The middle column shows the aligned binary mask obtained from the released plume-complex (PLM) raster. The right column shows the mask after dilation with a disk of radius six pixels. White denotes foreground in both mask columns.
- Model configurations and trained weights
- Patch identifiers and acquisition-level training, validation, and test split lists
- Data access and preprocessing instructions
