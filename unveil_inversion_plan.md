# Development Plan: Implementing Two-Stage Flow Inversion and AdaLN Control

This plan outlines the steps to modify `custom_flux_pipeline.py` to implement the image editing techniques described in the "Unveil Inversion and Invariance in Flow Transformer" paper, specifically the Two-Stage Flow Inversion and AdaLN-based Invariance Control using token-level features.

**Base Code:** `custom_flux_pipeline.py`
**Core Paper Algorithm:** Algorithm 1, Section 4.2, Section 5 (AdaLN Control)
**Target Transformer Module:** `diffusers.models.transformers.transformer_flux`
**Test Images:** `test_images/test_image_portrait.jpeg`, `test_images/test_image_house.jpeg`, `test_images/test_image_hand_artistic.png` (and their `.txt` prompts)

## Phase 1: Faithful Image Inversion

**Goal:** Implement and verify the ability to invert a source image to its initial noise latent (`x0`) and reconstruct it perfectly using the two-stage inversion process.

**Step 1.1: Implement Stage 1 (Fixed-Point Inversion Function)**

*   **Action:** Create a function `invert_image_fixed_point` in `custom_flux_pipeline.py` that takes a source image and optional prompt, performs fixed-point iteration (Eq. 9) with averaging (Alg 1, Line 8) over `I` iterations for `T` steps, and returns the initial latent `x0` and the full latent trajectory `[xT, ..., x0]`. Handle both prompted and unprompted inversion.
*   **Verification:**
    *   Run `invert_image_fixed_point` on all three test images (portrait, house, hand art) using parameters like `I=3`, `T=30`. Perform this **both prompted** (using their respective `.txt` files) **and unprompted**.
    *   Feed the resulting `x0` from each inversion into the *original* (unmodified generation loop) `run_custom_generation` function using the *source* prompt (if used for inversion).
    *   **Test:** Measure PSNR and LPIPS between each reconstructed image and its original source. Compare prompted vs. unprompted inversion results. Compare these metrics against a baseline using `I=1` (equivalent to plain Euler inversion). Expect significantly better reconstruction fidelity (higher PSNR, lower LPIPS) with `I > 1`, especially for prompted inversion.

**Step 1.2: Implement Stage 2 (Velocity Compensation Calculation)**

*   **Action:** Create a function `calculate_velocity_compensations` that takes the trajectory `[x0, ..., xT]` from Step 1.1 and the inversion prompt settings. It should loop forward from `t=0` to `T-1`, predict the velocity `v_pred` at each `x_t`, calculate the predicted next state `x_hat_{t+1}` (Eq. 10), and compute the compensation `epsilon_t = x_{t+1} - x_hat_{t+1}` (Eq. 11). Return the list of compensations `[eps_0, ..., eps_{T-1}]`.
*   **Verification:**
    *   Run `calculate_velocity_compensations` using the trajectories generated in Step 1.1 for all three test images (prompted and unprompted inversions).
    *   **Test:** Verify the output is a list of tensors with the correct length (`T`) and that each tensor has the same shape as the latent states. Check dtype consistency. Ensure the calculation runs without errors for all test cases.

**Step 1.3: Implement Reconstruction with Compensation**

*   **Action:** Modify the generation loop within a *new* function (e.g., `reconstruct_with_compensation`). Start generation from `x0` (from Step 1.1) using the source prompt (if any). In the loop, after predicting `v_pred = transformer(...)`, retrieve the corresponding `epsilon_t` (from Step 1.2). Calculate the compensated velocity `v_compensated = v_pred + epsilon_t / delta_sigma_t`. Use `v_compensated` in the scheduler step.
*   **Verification:**
    *   Run `reconstruct_with_compensation` using `x0` and `compensations` from the prompted inversions of all three test images.
    *   **Test:** Measure PSNR and LPIPS between each reconstructed image and its original source. Expect near-perfect reconstruction for all three images (e.g., PSNR > 40, LPIPS < 0.01, visually indistinguishable).

## Phase 2: AdaLN-Based Invariance Control for Editing

**Goal:** Modify the transformer and pipeline to enable text-guided editing using the calculated inversion, compensations, and token-level AdaLN feature swapping for invariance control.

**Step 2.1: Prepare Token-Level Control Inputs**

*   **Action:** Create helper functions/logic within the main editing script:
    1.  Tokenize source and target prompts.
    2.  Compute a boolean `diff_mask` indicating differing tokens between source/target.
    3.  Encode source prompt to get `encoder_hidden_states_src`, `pooled_projections_src`.
    4.  Encode target prompt to get `encoder_hidden_states_tgt`, `pooled_projections_tgt`.
*   **Verification:**
    *   **Test:** Use the test image prompts as source prompts and create target prompts for specific edits:
        *   Portrait: `"A photograph of a man with long hair in his 30s, soft smile"` -> `"A photograph of a man with long hair in his 30s, frowning"` (Target: `frowning`).
        *   House: `"A photograph of a soviet style residential building in early spring, during bright sunlight"` -> `"A photograph of a soviet style residential building in deep winter, covered in snow, during bright sunlight"` (Target: `deep winter, covered in snow`).
        *   Hand Art: `"..., a single, viscous stream of deep red blood trickles downward,..."` -> `"..., a single, viscous stream of bright blue liquid trickles downward,..."` (Target: `bright blue liquid`).
    *   Verify that the computed `diff_mask` correctly identifies *only* the target tokens (`frowning`, `deep winter, covered in snow`, `bright blue liquid`). Check shapes and dtypes of generated embeddings.

**Step 2.2: Modify `AdaLayerNormZero` and `FluxTransformerBlock`**

*   **Action:**
    1.  Modify `transformer_flux.AdaLayerNormZero.forward` signature and logic to accept `timestep`, `pooled_projections`, `encoder_hidden_states_src`, `encoder_hidden_states_tgt`, `diff_mask`, `is_context_norm`, `control_cutoff`.
    2.  Implement the `torch.where` logic inside `AdaLayerNormZero` to select source/target token embeddings based on `diff_mask` and `control_cutoff` *only* when `is_context_norm=False` and `timestep` is below cutoff. Pool the result and project to get `text_emb`. Combine with `time_emb` and `pooled_emb`. Compute modulation parameters. Add necessary projection layers (`self.text_embed`).
    3.  Modify `transformer_flux.FluxTransformerBlock.forward` signature to accept the new inputs and pass them correctly to the modified `self.norm1` and `self.norm1_context` (passing only target info to `norm1_context`).
*   **Verification:**
    *   **Test:** Create unit tests for the modified `AdaLayerNormZero.forward`. Provide mock inputs (including src/tgt embeddings mimicking the test cases from Step 2.1, masks, cutoffs). Assert that the intermediate `effective_encoder_hidden_states` uses source embeddings for masked positions when control is active and `is_context_norm=False`. Assert output shapes are correct. Ensure the logic for `is_context_norm=True` remains functional.

**Step 2.3: Implement Full Editing Pipeline with Token-Level Control**

*   **Action:** Create the main editing function `run_inversion_based_editing`. This function orchestrates:
    1.  Calling Step 1.1 (`invert_image_fixed_point`) - use prompted inversion for editing.
    2.  Calling Step 1.2 (`calculate_velocity_compensations`).
    3.  Preparing inputs from Step 2.1 (`diff_mask`, embeddings for source/target).
    4.  Running the generation loop, starting from `x0`.
    5.  Inside the loop: Call the modified `transformer` (from Step 2.2) passing all necessary arguments (`*_src`, `*_tgt` embeddings, `diff_mask`, `control_cutoff`, etc.).
    6.  Apply compensation `epsilon_t` to the predicted velocity `v_pred`.
    7.  Perform the scheduler step with the compensated velocity.
    8.  Decode the final image.
*   **Verification:**
    *   **Test:** Run end-to-end editing using the test images and the specific source -> target prompts defined in Step 2.1. Also try:
        *   Portrait: -> `"A photograph of a man with short hair in his 30s, soft smile"` (rigid change)
        *   House: -> `"A photograph of a soviet style residential building in early spring, at night time, illuminated by streetlights"` (attribute change)
        *   House: -> `"A photograph of a soviet style residential building in early spring, during bright sunlight, Van Gogh style"` (style change)
        *   Hand Art: -> `"..., cyberpunk illustration"` (style change)
    *   Use various `control_cutoff` values (e.g., 0.2, 0.4, 0.8) for each edit. Use prompted inversion results from Phase 1.
    *   Qualitatively evaluate: Does the edit succeed (frown appears, winter appears, liquid is blue, hair is short, lighting changes, style changes)? Are unedited regions (background, other features) preserved? Compare results across different `control_cutoff` values.
    *   Quantitatively evaluate (optional but recommended): Calculate CLIP score between the edited image and the *target* prompt. Mask the edited region (e.g., face for smile->frown, building for spring->winter, blood->liquid) and calculate PSNR/LPIPS/SSIM for the *unedited* region compared to the source image.

</rewritten_file> 