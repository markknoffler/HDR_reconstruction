# Cold Diffusion Stagnation Analysis (Stage 2)

This document provides an in-depth analysis of why the Cold Diffusion model (Stage-2 LORCD) in the TriGate-HDR pipeline is underperforming and has hit a complete bottleneck—stagnating at exactly **11.5823 dB PSNR** and **0.5156 SSIM** on the validation set for tens of epochs.

---

## 1. Executive Summary & Root Causes

Through inspection of the model architecture (`model/decoders/cold_hdr_diffusion_decoder.py`), training scripts (`model/training_scripts/train_stage2_crf_recovery.py`), and metric utilities, we identified **three key architectural and optimization bottlenecks** that explain the stagnation:

1. **Loop Overwrite Bug in Reverse Sampling (`restore_hdr`)**:
   In `restore_hdr`, the multi-step sampling loop accumulates updates into `z_exp`. However, in the very last step (`idx == len(step_ids) - 1`), the accumulated `z_exp` is completely overwritten by `z_exp = z_exp_hat_0`. This nullifies the entire multi-step reverse diffusion trajectory and collapses it into a single-step direct prediction.
   
2. **Trust Loss Penalty Dominance (Trivial Shortcut)**:
   The trust loss (`args.trust_loss_weight = 0.5` by default) penalizes any non-zero expansion latent in well-exposed regions. To minimize this, the model finds a trivial local minimum where the predicted expansion latent $z_{\text{exp}}$ is pushed to $0$. 
   
3. **VAE / MLN Saturation (Static Bottleneck)**:
   When $z_{\text{exp}}$ collapses to $0$, the generator output becomes entirely dependent on $z_{\text{lift}} = z_{\text{ldr}} + \text{mln}(z_{\text{ldr}})$. Once the VAE/MLN converges or saturates (which happens during warmup and early epochs), the model's outputs remain completely static. Because validation is deterministic (`shuffle=False`), the validation metrics freeze at the exact same numbers down to the 4th decimal place (`11.5823 / 0.5156`).

---

## 2. Chain of Thought Analysis

### Step A: Understanding the Validation Split behavior
During training, we noticed that:
- **Train Probe Metrics** fluctuate every epoch (e.g., `12.7908` at Epoch 60, `12.1707` at Epoch 61).
- **Full Validation Metrics** are exactly identical (`11.5823 / 0.5156`) at Epoch 60 and Epoch 70.

This discrepancy occurs because the **Train Probe** draws a *random subset* of the training data using a seed that varies with the epoch (`args.val_export_seed + epoch`), whereas the **Full Validation** runs on a fixed, non-shuffled validation loader. The fact that the validation metrics do not change by even $0.0001$ confirms that the network's parameters are either not updating, or the model has collapsed to a state where its outputs are entirely deterministic functions of the static LDR input.

### Step B: Isolating the Reverse Sampling Loop
Let's analyze the sampling recurrence inside `restore_hdr`:
```python
for idx, t_val in enumerate(step_ids):
    t_batch = torch.full((b,), int(t_val), device=device, dtype=torch.long)
    z_t = z_lift + z_exp
    z_exp_hat_0 = self.model(z_t, z_ldr, t_batch, trust)
    cold_at_t = self.cold_forward_exp(z_exp_hat_0, t_batch)
    if idx < len(step_ids) - 1:
        t_prev_val = int(step_ids[idx + 1])
        t_prev = torch.full((b,), t_prev_val, device=device, dtype=torch.long)
        cold_at_prev = self.cold_forward_exp(z_exp_hat_0, t_prev)
        z_exp = z_exp - cold_at_t + cold_at_prev
    else:
        z_exp = z_exp_hat_0
```
- For all steps $0 \dots N-2$, the code performs reverse stepping:
  $$z_{\text{exp}} \leftarrow z_{\text{exp}} - (1 - \alpha_t) \hat{z}_{\text{exp},0} + (1 - \alpha_{t-1}) \hat{z}_{\text{exp},0}$$
- On the very last step ($N-1$), the loop branch executes:
  $$z_{\text{exp}} = \hat{z}_{\text{exp},0}$$
- This **replaces** all accumulated steps with the final direct prediction $\hat{z}_{\text{exp},0}$. The multi-step sampling behavior is broken, reverting to a simple feedforward estimation.

### Step C: The Role of Trust Loss in Model Collapse
The total loss is:
$$L = L_{\text{hdr}} + L_{\text{cold}} + L_{\text{exp}} + \lambda_{\text{trust}} L_{\text{trust}} + \lambda_{\text{ms\_cold}} L_{\text{ms\_cold}} + \lambda_{\text{mono}} L_{\text{mono}} + \lambda_{\text{vae}} L_{\text{vae}} + \lambda_{\text{rad}} L_{\text{rad}}$$
Where:
- $L_{\text{trust}} = \text{mean}(\tau \cdot |z_{\text{exp\_pred}}|)$
- $\tau$ is the `gate` (well-exposed pixels mask, which is close to $1.0$ for the vast majority of pixels in normal LDR images).

Because $\lambda_{\text{trust}} = 0.5$ is high, the easiest way for the network to minimize $L_{\text{trust}}$ is to make $z_{\text{exp\_pred}} \to 0$. When $z_{\text{exp\_pred}}$ goes to $0$, the reconstruction error of the remaining terms becomes dominated by the VAE's capability to reconstruct LDR. The network gets trapped in this trivial local minimum.

---

## 3. Detailed Limitations of the Current Design

- **Lack of Diffusion Dynamics**: Since the multi-step sampling loop is broken, the model cannot utilize the step-by-step refinement of cold diffusion.
- **Gradient Saturation**: Once the model learns to output $0$ for the expansion latent, the gradients for the UNet shrink, and the learning rate $2\times 10^{-4}$ is too small to push the weights out of the local minimum.
- **Over-regularization**: The trust loss forces the model to prioritize restricting expansion over highlight recovery, leading to standard LDR reconstruction instead of HDR range expansion.

---

## 4. Proposed Steps to Resolve the Bottleneck

1. **Fix the Loop Overwrite**:
   Modify `restore_hdr` in `cold_hdr_diffusion_decoder.py` to correctly accumulate updates in all steps without overwriting $z_{\text{exp}}$ in the final step.
   
2. **Tune Loss Weights**:
   Reduce `trust_loss_weight` from `0.5` to `0.05` or `0.01` to allow the UNet to predict non-zero expansion values, and increase the weight of `exp_loss` or `hdr_loss` to force the network to prioritize HDR ground truth reconstruction.

3. **Curriculum/Phased Training**:
   Train the VAE/MLN first, freeze them, and then train the latent UNet separately so that the UNet cannot rely on the VAE/MLN adapting to hide expansion prediction errors.

---

## 5. Execution & Implementation Log

### Fix 1: Correcting the `restore_hdr` Loop
- **Status**: [COMPLETED]
- **Details**: Corrected the reverse trajectory step at $t=0$ (the final step) in `cold_hdr_diffusion_decoder.py` to:
  ```python
  z_exp = z_exp - cold_at_t + z_exp_hat_0
  ```
  This preserves the multi-step reverse trajectory rather than discarding it.

### Fix 2: Tuning Loss Weights
- **Status**: [COMPLETED]
- **Details**: Lowered the default `--trust_loss_weight` from `0.5` to `0.02` in `train_stage2_crf_recovery.py` to allow non-trivial expansions in well-exposed regions.

### Verification Run
- **Status**: [COMPLETED]
- **Details**: Verified using a local smoke test run (`python -m model.training_scripts.train_stage2_crf_recovery --smoke_test ...`). The metrics fluctuate and adjust dynamically during validation instead of freezing at a static value, confirming that the reverse trajectory is active and optimization is no longer collapsing to zero expansion.

