# U-Net Benchmark

**Revisiting model scaling with a U-net benchmark for 3D medical image segmentation**  
*Scientific Reports (2025)*

Paper: https://www.nature.com/articles/s41598-025-15617-1

## Overview

The script `add_sdw_variants_to_nnunetv2_plans.py` adds 18 custom U-Net configurations to nnU-Net v2 plans files. These configurations systematically vary three structural hyperparameters:

- **S ∈ {4, 5, 6}** → number of resolution stages
- **D ∈ {2, 3}** → convolution layers per stage  
- **W ∈ {16, 32, 64}** → initial channel width

This yields 18 variants (3 × 2 × 3) named as: `3d_fullres_S{S}D{D}W{W}`

## Environment Setup

This script is based on **nnU-Net v2.4.1** ([release link](https://github.com/MIC-DKFZ/nnUNet/releases/tag/v2.4.1)), which is the latest version as of August 2025.

**Note:** For other nnU-Net versions (e.g., nnU-Net v1, v2.2), the `nnUNetPlans.json` structure may differ slightly, requiring minor code adjustments.

Ensure the `nnUNet_preprocessed` environment variable is properly configured before running the script.

## Usage

1. **Plan and preprocess your dataset** (standard nnU-Net workflow):
   ```bash
   nnUNet_plan_and_preprocess -d DATASET_ID
   ```

2. **Add S/D/W variants to the plans**:
   ```bash
   python add_sdw_variants_to_nnunetv2_plans.py -d DATASET_ID
   ```

3. **Train models with the new configurations**:
   ```bash
   nnUNetv2_train DATASET_ID 3d_fullres_S{S}D{D}W{W} FOLD
   # e.g., nnUNetv2_train 5 3d_fullres_S4D2W16 0
   ```



## Citation

```bibtex
@article{huang2025unetbenchmark,
  title   = {Revisiting model scaling with a U-net benchmark for 3D medical image segmentation},
  author  = {Huang, Ziyan and Ye, Jin and Wang, Haoyu and Deng, Zhongying and Yang, Zhikai and Su, Yanzhou and Liu, Jie and Li, Tianbin and Gu, Yun and Zhang, Shaoting and Qiao, Yu and Gu, Lixu and He, Junjun},
  journal = {Scientific Reports},
  volume  = {15},
  number  = {29795},
  year    = {2025},
  doi     = {10.1038/s41598-025-15617-1}
}
```

## Acknowledgements

We are grateful to:
- The [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) team for the open-source codebase this project builds upon.
- The contributors of the 42 public [datasets](https://www.nature.com/articles/s41598-025-15617-1/tables/2) that enable open and reproducible research.
- Former members of the [GMAI](https://github.com/uni-medical) team who contributed to this project.


