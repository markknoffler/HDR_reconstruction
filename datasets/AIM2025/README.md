# AIM2025 Inverse Tone Mapping Dataset

ExpoCM uses the AIM 2025 ITM challenge data (~19,000 train pairs @ 256²).

## Manual download (registration required)

1. Register at https://www.codabench.org/competitions/8231/
2. Download the **development/training** bundle from the competition page.
3. Extract so this layout exists:

```
datasets/AIM2025/
  LDR_in/   # .jpg LDR inputs
  HDR_gt/   # .hdr or .exr ground truth
```

4. Re-run: `python scripts/download_expo_datasets.py --dataset aim2025 --prepare-only`
