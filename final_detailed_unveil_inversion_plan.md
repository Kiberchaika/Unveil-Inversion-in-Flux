# Detailed Development Plan: Implementing Unveil Inversion in Flux (v3)

This plan outlines the specific steps to modify `custom_flux_pipeline.py` (a script derived from the `diffusers` FluxPipeline) to implement the image editing techniques described in the "Unveil Inversion and Invariance in Flow Transformer" paper. It incorporates analysis of a previous RoPE implementation failure, provides detailed implementation guidance, and includes steps for saving intermediate inversion results.

**Target Paper:** Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing
**Base Code:** `custom_flux_pipeline.py` (or a copy)
**Target `diffusers` Components:** `FluxPipeline`, `FluxTransformer2DModel`, `diffusers.models.embeddings.apply_rotary_emb`, `diffusers.models.embeddings.ImagePositionalEmbeddings`, `FlowMatchEulerDiscreteScheduler`, `AdaLayerNormZero` (or equivalent modulation layers within the transformer blocks).
**Test Images:** `test_images/test_image_portrait.jpeg`, `test_images/test_image_house.jpeg`, `test_images/test_image_hand_artistic.png` (and their `.txt` prompts)
**Output Directories:** `output_custom_pipeline_inversion/`, `output_custom_pipeline_reconstruction/`, `output_custom_pipeline_editing/` (Create if they don't exist)

## Phase 1: Implement and Verify Patched RoPE for Image Embeddings

**Goal:** Create and verify a robust patch for `diffusers.models.embeddings.apply_rotary_emb` to correctly handle the 2D tuple `(cos, sin)` output from `ImagePositionalEmbeddings` when applied to the 4D query/key tensors within the Flux attention mechanism.

**Step 1.1: Implement Focused RoPE Debugging Script**

*   **Action:**
    1.  Create a copy of `custom_flux_pipeline.py` named `debug_rope.py` or modify the `if __name__ == "__main__":` block in the original file.
    2.  Inside `if __name__ == "__main__":`:
        *   Load the `FluxPipeline` for the chosen `model_id` and `dtype`.
        *   Extract necessary components: `pipeline.transformer`.
        *   Define the target device and dtype.
        *   Create a mock 4D query tensor `mock_x` with dimensions matching the previously observed error (e.g., `torch.randn(1, 24, 4608, 128, device=device, dtype=dtype)`). Use the known `num_heads` (24) and `head_dim` (128) for FLUX.1.
        *   Get the image positional embedding function: `image_pos_embed_func = pipeline.transformer.image_positional_embeddings`.
        *   Generate actual 2D image embeddings: `image_pos_embed_tuple = image_pos_embed_func(sequence_length=1536)` (Start with the sequence length `1536` that caused the mismatch `4608 vs 1536`).
        *   Import the original `apply_rotary_emb` from `diffusers.models.embeddings` and store it.
        *   Define the `_patched_apply_rotary_emb` function (initially it can just call the original or contain the logic from `custom_flux_pipeline_failed.py`).
        *   Globally apply the patch: `import diffusers.models.embeddings as embeddings_module; embeddings_module.apply_rotary_emb = _patched_apply_rotary_emb`.
        *   Wrap the call to the patched function in a `try...except RuntimeError as e:` block: `output = _patched_apply_rotary_emb(mock_x, image_pos_embed_tuple)`.
        *   Inside the `except` block, print the error message `e`.
        *   Add print statements inside `_patched_apply_rotary_emb` to log the shapes of `x`, `freqs_cos`, `freqs_sin`, `x_to_rotate`, `processed_freqs_cos`, `processed_freqs_sin` just before the call to `original_apply_rotary_emb` to aid debugging.
*   **Verification:** Running the script should initially reproduce the `RuntimeError: The size of tensor a (...) must match the size of tensor b (...)`.

*   **Result:**
    *   Created `debug_rope.py`.
    *   Identified that `transformer.image_positional_embeddings` was incorrect.
    *   Identified `diffusers.models.embeddings.ImagePositionalEmbeddings` was also incorrect for RoPE frequency generation.
    *   Correctly used `diffusers.models.embeddings.get_2d_rotary_pos_embed` to generate the `freqs_cis` tuple of shape `[seq_len, head_dim]`.
    *   Implemented a patched `_patched_apply_rotary_emb` that correctly handles sequence length mismatch between the 4D input tensor `x` and the 2D `freqs_cis` tuple by slicing `x`.
    *   Verified by running `debug_rope.py` that the patch successfully calls the original `apply_rotary_emb` without runtime errors when sequence lengths mismatch, achieving the goal of Step 1.1.

**Step 1.2: Refine `_patched_apply_rotary_emb` Logic**

*   **Action:** Modify the `if is_tuple_freqs:` block within `_patched_apply_rotary_emb` based on the following understanding: The `original_apply_rotary_emb` performs element-wise multiplication internally. The 2D `freqs_cos`/`sin` (`[image_seq_len, head_dim // 2]`) must be reshaped/broadcast to align with the 4D `x_to_rotate` (`[batch, heads, query_seq_len, head_dim]`).
    1.  **Reshape Embeddings:** Inside the `if is_tuple_freqs:` block, *before* sequence length slicing and *before* calling `original_apply_rotary_emb`:
        *   Reshape `processed_freqs_cos` from `[image_seq_len, head_dim // 2]` to `[1, 1, image_seq_len, head_dim // 2]`.
        *   Reshape `processed_freqs_sin` similarly.
        *   *(Self-Correction: The original function likely handles the broadcasting internally if the sequence lengths match. The primary issue is likely the sequence length mismatch when calling the original function, not the dimensionality *before* the call, although the patch needs to handle slicing correctly based on the *input* shapes. Furthermore, `get_2d_rotary_pos_embed` returns shape `[image_seq_len, head_dim]`, not `[image_seq_len, head_dim // 2]` when `use_real=True`)*
    2.  **Sequence Length Slicing:**
        *   Get `seq_len_x_patches = x.shape[2]` and `seq_len_freq_emb = freqs_cos.shape[0]`.
        *   If `seq_len_x_patches > seq_len_freq_emb`:
            *   Slice `x_to_rotate = x[:, :, :seq_len_freq_emb, :]`.
            *   Set `x_unrotated = x[:, :, seq_len_freq_emb:, :]`.
            *   Slice `processed_freqs_cos = freqs_cos[:seq_len_freq_emb, :]`.
            *   Slice `processed_freqs_sin = freqs_sin[:seq_len_freq_emb, :]`.
        *   If `seq_len_x_patches < seq_len_freq_emb`:
            *   Slice `processed_freqs_cos = freqs_cos[:seq_len_x_patches, :]`.
            *   Slice `processed_freqs_sin = freqs_sin[:seq_len_x_patches, :]`.
            *   `x_to_rotate` remains `x`, `x_unrotated` is `None`.
        *   Else (lengths match): `x_to_rotate` is `x`, `x_unrotated` is `None`, use full `processed_freqs_cos/sin`.
    3.  **Call Original Function:** Call `rotated_x = original_apply_rotary_emb(x=x_to_rotate, freqs_cis=(processed_freqs_cos, processed_freqs_sin))`.
    4.  **Concatenation:** Ensure the final concatenation `torch.cat((rotated_x, x_unrotated), dim=2)` uses `dim=2` (the sequence dimension for 4D tensors).
*   **Verification:** Run the debugging script from Step 1.1 iteratively after each refinement. The goal is to eliminate the `RuntimeError` related to tensor size mismatch when `original_apply_rotary_emb` is called with the processed tuple embeddings. The print statements should show compatible shapes being passed to the original function.
*   **Result:**
    *   Refined the `_patched_apply_rotary_emb` function in `debug_rope.py` to handle sequence length mismatches by slicing either the input tensor `x` or the frequency embeddings `freqs_cis` as needed.
    *   Verified by running `python debug_rope.py` that the refined patch successfully eliminates the `RuntimeError` when the sequence length of `x` (e.g., 4608) is greater than the sequence length of `freqs_cis` (e.g., 1536), achieving the goal of Step 1.2.
    *   No new files were added in this step; `debug_rope.py` was modified.

## Phase 2: Two-Stage Inversion Implementation and Verification

**Goal:** Implement and verify the two-stage inversion process (Algorithm 1 from the paper) using the verified RoPE patch, and save the inversion results (`x0` latent, compensations `ε`).

**Step 2.1: Implement Stage 1 (Fixed-Point Inversion Function)**

*   **Action:**
    1.  In `custom_flux_pipeline.py` (or the chosen working file), define the function `invert_image_fixed_point`:
        *   **Signature:** `(source_image_pil: Image.Image, prompt: Optional[str], pipeline: FluxPipeline, num_inversion_steps: int, num_inner_steps: int, guidance_scale: float = 0.0) -> Tuple[torch.Tensor, List[torch.Tensor]]`
        *   Get components from `pipeline`: `vae`, `text_encoder`, `tokenizer`, `text_encoder_2`, `tokenizer_2`, `transformer`, `scheduler`, `image_processor`.
        *   Set device and dtype from pipeline.
        *   Preprocess the source image: `image = pipeline.image_processor.preprocess(source_image_pil).to(device=device, dtype=dtype)`
        *   Encode the source image to target latents: `x_target_unpacked = pipeline.vae.encode(image).latent_dist.sample()` then `x_target_unpacked = x_target_unpacked * pipeline.vae.config.scaling_factor`, then `x_target_packed = _pack_latents(x_target_unpacked)`.
        *   Encode prompt (if provided) or get null embeddings:
            *   If `prompt`: `prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_custom(...)`
            *   If `prompt is None`: Generate null embeddings (requires adapting `encode_prompt_custom` or using pipeline's internal logic for null prompts). For simplicity initially, focus on prompted inversion.
        *   Calculate sigmas: `sigmas = np.linspace(1.0, scheduler.config.sigma_min, num_inversion_steps + 1)` (or use scheduler's internal logic if available).
        *   Set scheduler timesteps: `scheduler.set_timesteps(sigmas=sigmas, device=device)`. Print `len(scheduler.timesteps)` to confirm `T+1` steps.
        *   Initialize `latents = x_target_packed.clone()` (this is $x_{T+1}$ conceptually, index T).
        *   Initialize `latent_trajectory = [latents.cpu()]`.
        *   Loop backward: `for t in tqdm(reversed(scheduler.timesteps[:-1]), total=num_inversion_steps, desc="Fixed-Point Inversion")`: (Loops T times, from index T-1 down to 0)
            *   Get `step_index = (scheduler.timesteps == t).nonzero()[0].item()`.
            *   Get `sigma_t = scheduler.sigmas[step_index]` and `sigma_t_plus_1 = scheduler.sigmas[step_index + 1]`.
            *   Initialize `x_prev_accum = torch.zeros_like(latents)`.
            *   Initialize `x_t_estim_inner = latents.clone()` (represents $x_t$ estimate for inner loop, starts as $x_{t+1}$).
            *   Inner loop: `for _ in range(num_inner_steps)`:
                *   Prepare transformer inputs using `x_t_estim_inner`, `prompt_embeds`, `pooled_prompt_embeds`, `text_ids`, `latent_image_ids` (prepare these once outside the main loop), `t` (as tensor), `guidance_scale` (if applicable).
                *   Predict velocity: `velocity = pipeline.transformer(**transformer_inputs).sample`.
                *   Calculate previous state estimate $x_{t-1}^{\text{estim}}$ using `scheduler.step(velocity, t, x_t_estim_inner)` (or equivalent logic if step requires sigmas).
                *   Accumulate: `x_prev_accum += x_{t-1}^{\text{estim}}`.
                *   Refine $x_t$ estimate for *next* inner iteration using Eq. 9: `x_t_estim_inner = latents + (sigma_t - sigma_t_plus_1) * velocity`.
            *   Average accumulated previous states: `x_prev_final = x_prev_accum / num_inner_steps`.
            *   Update for next outer step: `latents = x_prev_final`.
            *   Store trajectory: `latent_trajectory.append(latents.cpu())`.
        *   Reverse trajectory: `latent_trajectory.reverse()` # Now it's [x0, x1, ..., xT]
        *   Return `latents` (final `x0`) and `latent_trajectory`.
*   **Verification:** Run the function with test images, `T=30`, `I=3`. Check output shapes and dtypes. Ensure it runs without errors.

**Step 2.2: Implement Reconstruction & Metrics**

*   **Action:**
    1.  Implement `run_custom_generation_from_latents`:
        *   **Signature:** `(latents_packed_x0: torch.Tensor, prompt: str, pipeline: FluxPipeline, num_inference_steps: int, guidance_scale: float, seed: int) -> Image.Image`
        *   Get components, device, dtype.
        *   Encode prompt.
        *   Prepare `latent_image_ids`.
        *   Set scheduler timesteps for *generation* (forward process).
        *   Initialize `latents = latents_packed_x0.to(device=device, dtype=dtype)`.
        *   Run the standard forward generation loop (similar to `run_custom_generation` but starting from `latents_packed_x0` and using the generation timesteps/sigmas).
        *   Unpack final latents: `latents_unpacked = _unpack_latents(...)`.
        *   Decode image: `image = pipeline.vae.decode(latents_unpacked / pipeline.vae.config.scaling_factor).sample`.
        *   Post-process and return PIL image.
    2.  Implement `calculate_metrics(img1_pil: Image.Image, img2_pil: Image.Image, device: torch.device)`:
        *   Use `lpips` library (needs installation) and `torchmetrics.PeakSignalNoiseRatio`.
        *   Preprocess PIL images to tensors suitable for the metrics.
        *   Calculate and return PSNR and LPIPS values.
*   **Verification:** Run `run_custom_generation_from_latents` with manually created noise latents to ensure the forward pass works. Run `calculate_metrics` with identical and different images to check outputs.

**Step 2.3: Verification 1 & Save x0/Trajectory**

*   **Action:**
    1.  Modify the main script (`if __name__ == "__main__":`).
    2.  Load test images and corresponding prompts.
    3.  For each test image/prompt:
        *   Run `invert_image_fixed_point` with `I=3` (prompted). Store `x0_latent_i3`, `trajectory_i3`.
        *   Run `invert_image_fixed_point` with `I=1` (prompted). Store `x0_latent_i1`.
        *   Run `invert_image_fixed_point` with `I=3` (unprompted, pass `prompt=None`). Store `x0_latent_i3_unprompted`.
    4.  **Save Results:**
        *   `torch.save(x0_latent_i3.cpu(), f"output_custom_pipeline_inversion/{image_name}_prompted_I3_x0.pt")`
        *   `torch.save(trajectory_i3, f"output_custom_pipeline_inversion/{image_name}_prompted_I3_traj.pt")` (Trajectory is already on CPU)
    5.  Run reconstruction using `run_custom_generation_from_latents`:
        *   `recon_i3 = run_custom_generation_from_latents(x0_latent_i3, source_prompt, ...)`
        *   `recon_i1 = run_custom_generation_from_latents(x0_latent_i1, source_prompt, ...)`
        *   `recon_i3_unprompted = run_custom_generation_from_latents(x0_latent_i3_unprompted, source_prompt, ...)`
    6.  Save reconstructed images (e.g., `output_custom_pipeline_reconstruction/{image_name}_recon_I3.png`).
    7.  Calculate metrics: `metrics_i3 = calculate_metrics(source_image_pil, recon_i3, ...)` etc.
    8.  Print and compare metrics (PSNR, LPIPS) for I=3 vs I=1 vs I=3_unprompted.
*   **Verification:** Metrics for `I=3` prompted inversion should show significantly better reconstruction fidelity (higher PSNR, lower LPIPS) compared to `I=1` and unprompted `I=3`. Reconstructed images should visually match the source closely for `I=3` prompted.

**Step 2.4: Implement Stage 2 (Velocity Compensation Calculation)**

*   **Action:**
    1.  Define function `calculate_velocity_compensations`:
        *   **Signature:** `(latent_trajectory: List[torch.Tensor], prompt: Optional[str], pipeline: FluxPipeline, guidance_scale: float = 0.0) -> List[torch.Tensor]`
        *   Get components, device, dtype.
        *   Load trajectory to the correct device. Trajectory is `[x0, x1, ..., xT]`.
        *   Encode prompt or get null embeddings.
        *   Prepare `latent_image_ids`.
        *   Set scheduler timesteps matching the *inversion* process.
        *   Initialize `compensations = []`.
        *   Loop forward: `for t_idx in tqdm(range(len(scheduler.timesteps) - 1), desc="Calculating Compensations")`: (Loops T times, index 0 to T-1)
            *   Get current latent `x_t = latent_trajectory[t_idx].to(device=device, dtype=dtype)`.
            *   Get next latent from trajectory `x_t_plus_1_actual = latent_trajectory[t_idx + 1].to(device=device, dtype=dtype)`.
            *   Get current timestep `t = scheduler.timesteps[t_idx]`.
            *   Get sigmas: `sigma_t = scheduler.sigmas[t_idx]`, `sigma_t_plus_1 = scheduler.sigmas[t_idx + 1]`.
            *   Prepare transformer inputs using `x_t`, `prompt_embeds`, etc., and `t`.
            *   Predict velocity: `velocity = pipeline.transformer(**transformer_inputs).sample`.
            *   Calculate predicted next state using Eq. 10: `x_t_plus_1_predicted = x_t + (sigma_t_plus_1 - sigma_t) * velocity`.
            *   Calculate compensation using Eq. 11: `epsilon_t = x_t_plus_1_actual - x_t_plus_1_predicted`.
            *   Store: `compensations.append(epsilon_t.cpu())`.
        *   Return `compensations` list `[ε_0, ..., ε_{T-1}]`.
*   **Verification:** Run the function using saved trajectories. Check output list length (`T`) and tensor shapes/dtypes. Ensure it runs without errors.

**Step 2.5: Verification 2 & Save Compensations**

*   **Action:**
    1.  In the main script, after generating/saving trajectories for prompted `I=3`:
        *   Load the trajectory if not in memory.
        *   Run `compensations = calculate_velocity_compensations(trajectory_i3, source_prompt, ...)`
    2.  **Save Results:**
        *   `torch.save(compensations, f"output_custom_pipeline_inversion/{image_name}_prompted_I3_eps.pt")`
*   **Verification:** Check that the saved file exists and can be loaded.

**Step 2.6: Implement Reconstruction with Compensation**

*   **Action:**
    1.  Define function `reconstruct_with_compensation`:
        *   **Signature:** `(x0_latent: torch.Tensor, compensations: List[torch.Tensor], prompt: str, pipeline: FluxPipeline, num_inference_steps: int, guidance_scale: float, seed: int) -> Image.Image`
        *   Get components, device, dtype.
        *   Load `x0_latent` and `compensations` to device.
        *   Encode prompt.
        *   Prepare `latent_image_ids`.
        *   Set scheduler timesteps for *generation* (must match the number of compensation steps, `T`).
        *   Initialize `latents = x0_latent.to(device=device, dtype=dtype)`.
        *   Run the forward generation loop: `for i, t in tqdm(enumerate(scheduler.timesteps), total=num_inference_steps)`:
            *   Get sigmas: `sigma_t = scheduler.sigmas[i]`, `sigma_t_plus_1 = scheduler.sigmas[i + 1]`.
            *   Prepare transformer inputs using `latents`, `prompt_embeds`, etc., and `t`.
            *   Predict velocity: `v_pred = pipeline.transformer(**transformer_inputs).sample`.
            *   Get compensation: `epsilon_t = compensations[i].to(device=device, dtype=dtype)`.
            *   Calculate compensated velocity: `v_compensated = v_pred + epsilon_t / (sigma_t_plus_1 - sigma_t)`.
            *   Perform scheduler step: `latents = scheduler.step(v_compensated, t, latents).prev_sample`.
        *   Unpack final latents.
        *   Decode image.
        *   Post-process and return PIL image.
*   **Verification:** Run the function with test data. Check shapes and dtypes during execution.

**Step 2.7: Verification 3 (Perfect Reconstruction)**

*   **Action:**
    1.  In the main script:
        *   Load saved `x0_latent` and `compensations` for prompted `I=3` inversions.
        *   Run `recon_compensated = reconstruct_with_compensation(x0_latent, compensations, source_prompt, ...)`
    2.  Save compensated reconstructed images (e.g., `output_custom_pipeline_reconstruction/{image_name}_recon_compensated.png`).
    3.  Calculate metrics: `metrics_compensated = calculate_metrics(source_image_pil, recon_compensated, ...)`
    4.  Print and compare metrics (PSNR, LPIPS) with previous reconstructions.
*   **Verification:** Expect near-perfect reconstruction metrics (PSNR > 40, LPIPS < 0.01). Visually compare `recon_compensated` with the original source image.

## Phase 3: AdaLN-Based Invariance Control Implementation and Verification

**Goal:** Modify the transformer and pipeline to enable text-guided editing using the calculated inversion results (`x0`, `ε`) and token-level AdaLN feature swapping for invariance control (Paper Section 5).

**Step 3.1: Prepare Token-Level Control Inputs**

*   **Action:**
    1.  Define a helper function `prepare_control_inputs(source_prompt, target_prompt, pipeline, device, dtype, max_sequence_length)`:
        *   Tokenize source prompt `P_s` using `pipeline.tokenizer_2`.
        *   Tokenize target prompt `P_t` using `pipeline.tokenizer_2`.
        *   Identify differing token indices between `P_s` and `P_t` (handle padding and special tokens). Create a boolean `diff_mask` of shape `[1, max_sequence_length]`.
        *   Encode `P_s` using `encode_prompt_custom` to get `prompt_embeds_src`, `pooled_prompt_embeds_src`, `text_ids_src`.
        *   Encode `P_t` using `encode_prompt_custom` to get `prompt_embeds_tgt`, `pooled_prompt_embeds_tgt`, `text_ids_tgt`.
        *   Return `diff_mask`, `prompt_embeds_src`, `pooled_prompt_embeds_src`, `prompt_embeds_tgt`, `pooled_prompt_embeds_tgt`.
*   **Verification:**
    *   Test with specific source -> target prompt pairs:
        *   Portrait: `"A photograph of a man with long hair in his 30s, soft smile"` -> `"A photograph of a man with long hair in his 30s, frowning"`
        *   House: `"A photograph of a soviet style residential building in early spring, during bright sunlight"` -> `"A photograph of a soviet style residential building in deep winter, covered in snow, during bright sunlight"`
        *   Hand Art: `"..., a single, viscous stream of deep red blood trickles downward,..."` -> `"..., a single, viscous stream of bright blue liquid trickles downward,..."`
    *   Verify that the computed `diff_mask` correctly identifies *only* the target token indices. Check shapes and dtypes of generated embeddings.

**Step 3.2: Modify Transformer for AdaLN Control**

*   **Action:**
    1.  **Analysis:** Locate the `AdaLayerNormZero` class (or equivalent modulation mechanism like `FluxModulation`) within the `diffusers` implementation of `FluxTransformer2DModel` (likely in `diffusers.models.transformers.transformer_flux`). Understand how `timestep`, `encoder_hidden_states` (token-level), and `pooled_projections` are combined to compute `shift`, `scale`, `gate`.
    2.  **Modification:**
        *   Modify the `forward` signature of the identified modulation layer(s) (e.g., `AdaLayerNormZero.forward`) to accept additional arguments: `encoder_hidden_states_src`, `encoder_hidden_states_tgt`, `diff_mask`, `control_cutoff`. Make these optional (`None`) to maintain compatibility with standard generation.
        *   Inside the `forward` method, *before* the token-level `encoder_hidden_states` are used (e.g., potentially pooled or directly used in calculating modulation):
            *   Check if `encoder_hidden_states_src` is not `None` (indicating control is active).
            *   Get the current timestep value (it might be passed directly or derived from `timestep` embedding).
            *   Compare timestep with `control_cutoff`.
            *   If control is active and `timestep < control_cutoff`:
                *   `effective_hidden_states = torch.where(diff_mask.unsqueeze(-1), encoder_hidden_states_tgt, encoder_hidden_states_src)`
            *   Else:
                *   `effective_hidden_states = encoder_hidden_states_tgt` (or the original `encoder_hidden_states` passed in).
            *   Use `effective_hidden_states` in the subsequent calculations instead of the original `encoder_hidden_states`.
        *   Modify the `forward` signature of the parent block(s) (e.g., `FluxTransformerBlock`) and the main `FluxTransformer2DModel.forward` to accept and pass these new arguments down.

*   **Verification:** This step is complex to verify in isolation without running the model. Rely on careful implementation and the unit tests in the next step.

**Step 3.3: Verification (Unit Tests)**

*   **Action:**
    1.  Create unit tests specifically for the modified modulation layer(s) (e.g., `TestModifiedAdaLayerNormZero`).
    2.  Instantiate the layer.
    3.  Create mock inputs: `timestep` tensor, `pooled_projections`, `encoder_hidden_states_src`, `encoder_hidden_states_tgt`, `diff_mask`. Ensure shapes and dtypes are correct.
    4.  Call the modified `forward` method with different `control_cutoff` values and timesteps (above and below the cutoff).
    5.  Assert that the calculated `shift`, `scale`, `gate` (or intermediate values derived from `effective_hidden_states`) reflect the use of source embeddings for masked tokens when control is active and the timestep is below the cutoff, and target embeddings otherwise.
*   **Verification:** Unit tests pass, confirming the conditional logic works as expected.

**Step 3.4: Implement Full Editing Pipeline**

*   **Action:**
    1.  Define the main editing function `run_inversion_based_editing`:
        *   **Signature:** `(source_image_pil: Image.Image, source_prompt: str, target_prompt: str, pipeline: FluxPipeline, num_inversion_steps: int, num_inner_steps: int, control_cutoff: float, guidance_scale: float = 0.0, seed: int = 42) -> Image.Image`
        *   Get components, device, dtype.
        *   **Load/Run Inversion:**
            *   Option 1 (Run): Call `invert_image_fixed_point` with `source_image_pil`, `source_prompt`, `I=num_inner_steps`, `T=num_inversion_steps`. Get `x0_latent`, `trajectory`.
            *   Option 2 (Load): Load pre-saved `x0_latent` and `trajectory` from Phase 2.
        *   **Load/Run Compensation:**
            *   Option 1 (Run): Call `calculate_velocity_compensations` with `trajectory`, `source_prompt`. Get `compensations`.
            *   Option 2 (Load): Load pre-saved `compensations` from Phase 2.
        *   **Prepare Control Inputs:** Call `prepare_control_inputs(source_prompt, target_prompt, ...)` to get `diff_mask`, `prompt_embeds_src`, `pooled_prompt_embeds_src`, `prompt_embeds_tgt`, `pooled_prompt_embeds_tgt`.
        *   Prepare `latent_image_ids`.
        *   Set scheduler timesteps for generation (`T` steps).
        *   Initialize `latents = x0_latent.to(device=device, dtype=dtype)`.
        *   Run the generation loop: `for i, t in tqdm(enumerate(scheduler.timesteps), total=num_inference_steps)`:
            *   Get sigmas.
            *   Prepare transformer inputs: `latents`, `t`, `latent_image_ids`, and crucially, pass `encoder_hidden_states=prompt_embeds_tgt`, `pooled_projections=pooled_prompt_embeds_tgt`, **plus the new control args:** `encoder_hidden_states_src=prompt_embeds_src`, `diff_mask=diff_mask`, `control_cutoff=control_cutoff`.
            *   Predict velocity: `v_pred = pipeline.transformer(**transformer_inputs).sample`.
            *   Get compensation: `epsilon_t = compensations[i].to(device=device, dtype=dtype)`.
            *   Calculate compensated velocity: `v_compensated = v_pred + epsilon_t / (sigma_t_plus_1 - sigma_t)`.
            *   Perform scheduler step: `latents = scheduler.step(v_compensated, t, latents).prev_sample`.
        *   Unpack final latents.
        *   Decode image.
        *   Post-process and return PIL image.
*   **Verification:** Ensure the function signature is correct and all necessary arguments are passed to the modified transformer.

**Step 3.5: Verification (End-to-End Editing)**

*   **Action:**
    1.  In the main script:
        *   Use the test images and the specific source -> target prompts defined in Step 3.1.
        *   Also try additional edits:
            *   Portrait: -> `"A photograph of a man with short hair in his 30s, soft smile"`
            *   House: -> `"A photograph of a soviet style residential building in early spring, at night time, illuminated by streetlights"`
            *   House: -> `"A photograph of a soviet style residential building in early spring, during bright sunlight, Van Gogh style"`
            *   Hand Art: -> `"..., cyberpunk illustration"`
        *   Run `run_inversion_based_editing` for each case, using the saved/recalculated inversion results (prompted, `I=3`).
        *   Experiment with various `control_cutoff` values (e.g., 0.2, 0.4, 0.8).
        *   Save edited images (e.g., `output_custom_pipeline_editing/{image_name}_{target_edit}_cutoff{cutoff}.png`).
    2.  **Qualitative Evaluation:** Visually inspect the edited images. Does the edit succeed? Are unedited regions preserved? How does `control_cutoff` influence the balance between editing and preservation?
    3.  **Quantitative Evaluation (Optional):**
        *   Calculate CLIP score between the edited image and the *target* prompt.
        *   Manually create masks for non-edited regions if possible. Calculate PSNR/LPIPS/SSIM for the *unedited* region compared to the source image.
*   **Verification:** The editing process completes successfully. Qualitative results show successful edits with reasonable preservation of unedited areas, controllable via `control_cutoff`. Quantitative metrics (if used) align with qualitative observations.