import torch
import diffusers # Ensure diffusers is imported before patching
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
# Importing necessary components and utilities directly if possible, otherwise copy/adapt
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin # Needed for prompt conversion check
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
from tqdm import tqdm
from typing import Tuple # Already imported List, Optional, Union, etc.
from diffusers.models import AutoencoderKL # Import the VAE class
# Store original RoPE function before patching
from diffusers.models.embeddings import apply_rotary_emb as _original_apply_rotary_emb

# --- Configuration ---
model_id = "black-forest-labs/FLUX.1-schnell"
# Use float16 for memory efficiency as suggested in the video
dtype = torch.float16
output_dir = "output_custom_pipeline"
prompt = "Roman Empire soldier with a sword and a shield, horses in background, mountain with snow in background"
seed = 42

logger = logging.get_logger(__name__)

# --- Patched RoPE Function (from debug_rope.py) ---
def _patched_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    use_real: bool = False, # Add use_real parameter explicitly
    **kwargs
) -> torch.Tensor:
    """
    Patched version to handle potential tuple freqs_cis from ImagePositionalEmbeddings,
    slicing x if its sequence length is longer than the embedding sequence length.
    """
    is_tuple_freqs = isinstance(freqs_cis, tuple)
    # print(f"--- Inside _patched_apply_rotary_emb ---") # Reduce verbosity
    # print(f"  x shape: {x.shape}")
    # print(f"  freqs_cis type: {type(freqs_cis)}")

    if is_tuple_freqs:
        freqs_cos, freqs_sin = freqs_cis
        # print(f"  freqs_cos shape: {freqs_cos.shape}")
        # print(f"  freqs_sin shape: {freqs_sin.shape}")

        seq_len_x = x.shape[2] # Input query/key sequence length
        seq_len_freq = freqs_cos.shape[0] # Embedding sequence length
        # print(f"  Input x seq len: {seq_len_x}, Freq seq len: {seq_len_freq}")

        # Sequence Length Handling
        x_to_rotate = x
        x_unrotated = None
        processed_freqs_cos = freqs_cos
        processed_freqs_sin = freqs_sin

        if seq_len_x > seq_len_freq:
            # print(f"  Input x seq len ({seq_len_x}) > Freq seq len ({seq_len_freq}). Slicing x.")
            x_to_rotate = x[:, :, :seq_len_freq, :]
            x_unrotated = x[:, :, seq_len_freq:, :]
            # Use full frequency tensors
            processed_freqs_cos = freqs_cos
            processed_freqs_sin = freqs_sin
        elif seq_len_x < seq_len_freq:
            # print(f"  Input x seq len ({seq_len_x}) < Freq seq len ({seq_len_freq}). Slicing freqs.")
            processed_freqs_cos = freqs_cos[:seq_len_x, :]
            processed_freqs_sin = freqs_sin[:seq_len_x, :]
            x_to_rotate = x
            x_unrotated = None
        # else: # Lengths match, no slicing needed
        #    print("  Input x seq len matches Freq seq len. No slicing needed.")

        # Call original function with potentially sliced x and full freqs
        # Pass use_real=True as the tuple format implies real components
        rotated_x = _original_apply_rotary_emb(
            x=x_to_rotate,
            freqs_cis=(processed_freqs_cos, processed_freqs_sin),
            use_real=True
        )

        # Concatenate unrotated part if necessary
        if x_unrotated is not None:
            output = torch.cat((rotated_x, x_unrotated), dim=2)
        else:
            output = rotated_x

    else:
        # Handle standard tensor freqs_cis (pass through)
        # print(f"  freqs_cis shape: {freqs_cis.shape}")
        # print("  Calling original apply_rotary_emb with tensor...")
        output = _original_apply_rotary_emb(x=x, freqs_cis=freqs_cis, use_real=use_real, **kwargs)

    # print(f"--- Exiting _patched_apply_rotary_emb --- Output shape: {output.shape}")
    return output

# Apply the patch globally
print("Applying global RoPE patch...")
diffusers.models.embeddings.apply_rotary_emb = _patched_apply_rotary_emb
print("Global RoPE patch applied.")

# --- Helper functions (Copied/Adapted from diffusers.pipelines.flux.pipeline_flux) ---

#calculate_shift is simple enough to copy
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# retrieve_timesteps is also relatively standalone
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Need to adapt prompt encoding logic. We won't inherit from Mixins, so simplify.
def _get_t5_prompt_embeds_custom(
    prompt: Union[str, List[str]],
    num_images_per_prompt: int,
    max_sequence_length: int,
    device: torch.device,
    tokenizer_2, # Pass in component
    text_encoder_2 # Pass in component
):
    dtype = text_encoder_2.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # Simplified: Assume no textual inversion for this custom pipeline
    # if isinstance(self, TextualInversionLoaderMixin):
    #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

    text_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    # Warning for truncation (simplified)
    # ...

    prompt_embeds = text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds

def _get_clip_prompt_embeds_custom(
    prompt: Union[str, List[str]],
    num_images_per_prompt: int,
    device: torch.device,
    tokenizer, # Pass in component
    text_encoder, # Pass in component
    tokenizer_max_length: int
):
    dtype = text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # Simplified: Assume no textual inversion
    # if isinstance(self, TextualInversionLoaderMixin):
    #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    # Warning for truncation (simplified)
    # ...

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    pooled_prompt_embeds = prompt_embeds.pooler_output # Use pooled output
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype, device=device)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return pooled_prompt_embeds

def encode_prompt_custom(
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]],
    device: torch.device,
    num_images_per_prompt: int,
    max_sequence_length: int,
    # Pass components
    text_encoder,
    tokenizer,
    text_encoder_2,
    tokenizer_2,
    tokenizer_max_length,
    # Ignore LoRA scale for simplicity now
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = prompt_2 or prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    pooled_prompt_embeds = _get_clip_prompt_embeds_custom(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_max_length=tokenizer_max_length
    )
    prompt_embeds = _get_t5_prompt_embeds_custom(
        prompt=prompt_2,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2
    )

    # text_ids are usually just zeros for standard generation
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype) # Match dtype

    return prompt_embeds, pooled_prompt_embeds, text_ids


# Latent helpers
def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    # height and width here are latent height/width AFTER 8x VAE reduction AND 2x packing
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    # Repeat for batch size - this was missing in original thought, check pipeline code again
    # The original code doesn't seem to repeat for batch size here, the transformer might handle broadcasting? Let's assume that for now.
    return latent_image_ids.to(device=device, dtype=dtype)

def _pack_latents(latents):
    # Assumes latents shape: (batch_size, num_channels_latents, height, width)
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5) # b h/2 w/2 c 2 2
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4) # b N C*4
    return latents

def _unpack_latents(latents, target_height, target_width, vae_scale_factor):
    # latents shape: (batch_size, num_patches, channels) = (b, N, C*4)
    batch_size, num_patches, packed_channels = latents.shape
    channels = packed_channels // 4

    # Calculate the unpacked latent dimensions
    # Target height/width are the *original image* dimensions
    latent_height = target_height // vae_scale_factor
    latent_width = target_width // vae_scale_factor

    # num_patches should be (latent_height // 2) * (latent_width // 2)
    height_div_2 = latent_height // 2
    width_div_2 = latent_width // 2

    if num_patches != height_div_2 * width_div_2:
         # This indicates a mismatch, recalculate based on num_patches if possible, or raise error
         # For simplicity, let's assume the input target_height/width were correct for now
         pass


    latents = latents.view(batch_size, height_div_2, width_div_2, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5) # b c h/2 2 w/2 2
    latents = latents.reshape(batch_size, channels, latent_height, latent_width) # b c H W (latent)
    return latents

def prepare_latents_custom(
    batch_size,
    num_channels_latents,
    height, # Target image height
    width, # Target image width
    dtype,
    device,
    generator,
    vae_scale_factor, # Pass this in
    latents=None, # Allow passing latents
):
    # Calculate latent dimensions based on VAE scale factor AND packing factor (2)
    latent_height = height // (vae_scale_factor * 2)
    latent_width = width // (vae_scale_factor * 2)

    shape = (batch_size, num_channels_latents, latent_height * 2, latent_width * 2) # Shape *before* packing

    if latents is not None:
        # Assume passed latents are already packed correctly if not None
        # Need latent_image_ids regardless
        latent_image_ids = _prepare_latent_image_ids(batch_size, latent_height, latent_width, device, dtype)
        return latents.to(device=device, dtype=dtype), latent_image_ids

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError("Batch size mismatch with generator list length.")

    # Generate noise in the unpacked shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    # Pack the noise
    latents = _pack_latents(noise)

    latent_image_ids = _prepare_latent_image_ids(batch_size, latent_height, latent_width, device, dtype)

    return latents, latent_image_ids


# --- Main Custom Generation Function ---
@torch.no_grad()
def run_custom_generation(
    vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, transformer, scheduler, image_processor, # Components
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float, # Note: Only used if guidance_embeds=True
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
    # Add other relevant params from __call__ if needed (e.g., negative_prompt)
):
    print("--- Starting Custom Generation Function ---")
    # 1. Define call parameters
    batch_size = 1 # Assuming single prompt for now
    num_images_per_prompt = 1
    # Simplification: Ignore Classifier-Free Guidance for this initial extraction
    do_true_cfg = False
    negative_prompt = None
    negative_prompt_2 = None

    # 2. Encode prompt
    print("Encoding prompt...")
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_custom(
        prompt=prompt,
        prompt_2=prompt, # Use same prompt for both encoders
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        # Pass components
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        tokenizer_max_length=tokenizer.model_max_length,
    )
    print(f"  Prompt Embeds shape: {prompt_embeds.shape}, Pooled shape: {pooled_prompt_embeds.shape}")

    # 3. Prepare latent variables
    print("Preparing latents...")
    # Note: in_channels for transformer is for *packed* latents (C*4)
    # num_channels_latents should be channels *before* packing
    num_channels_latents = transformer.config.in_channels // 4
    generator = torch.Generator(device="cpu").manual_seed(seed) # Use CPU generator

    # Pass the VAE scale factor needed by prepare_latents_custom
    # The VaeImageProcessor stores the base factor (e.g., 8), adjust for packing
    vae_scale_factor_for_latents = image_processor.vae_scale_factor

    latents, latent_image_ids = prepare_latents_custom(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        dtype,
        device, # Generate latents directly on the target device if possible
        generator,
        vae_scale_factor=vae_scale_factor_for_latents
    )
    print(f"  Latents shape (packed): {latents.shape}, Latent IDs shape: {latent_image_ids.shape}")


    # 4. Prepare timesteps
    print("Preparing timesteps...")
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1] # Sequence length of packed latents
    # Use scheduler's config values if available, otherwise use defaults from original code
    base_seq_len = getattr(scheduler.config, "base_image_seq_len", 256)
    max_seq_len_cfg = getattr(scheduler.config, "max_image_seq_len", 4096)
    base_shift_cfg = getattr(scheduler.config, "base_shift", 0.5)
    max_shift_cfg = getattr(scheduler.config, "max_shift", 1.16)

    mu = calculate_shift(
        image_seq_len, base_seq_len, max_seq_len_cfg, base_shift_cfg, max_shift_cfg
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    print(f"  Using {num_inference_steps} inference steps.")

    # 5. Prepare guidance embeds if needed
    guidance = None
    if getattr(transformer.config, "guidance_embeds", False): # Check if config exists
         guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32) # Use float32 for guidance
         guidance = guidance.expand(latents.shape[0])
         print("  Guidance embedding enabled.")

    # 6. Denoising loop
    print("Starting custom denoising loop...")
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
    # Use progress bar similar to pipeline? Requires tqdm. Skip for now.
    for i, t in enumerate(timesteps):
        if i < num_warmup_steps: # Skip warmup steps if any for progress reporting
             continue
        print(f"  Step {i+1-num_warmup_steps}/{num_inference_steps}, Timestep: {t.item():.4f}")

        # broadcast to batch dimension
        timestep = t.expand(latents.shape[0]).to(latents.dtype) # Match latent dtype

        # predict the noise residual
        # Ensure inputs match transformer expected dtypes
        noise_pred = transformer(
            hidden_states=latents.to(transformer.dtype),
            timestep=(timestep / 1000).to(transformer.dtype), # Scale timestep
            guidance=guidance.to(transformer.dtype) if guidance is not None else None,
            pooled_projections=pooled_prompt_embeds.to(transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(transformer.dtype),
            txt_ids=text_ids.to(transformer.dtype),
            img_ids=latent_image_ids.to(transformer.dtype),
            # joint_attention_kwargs=None, # Keep simple
            return_dict=False,
        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Explicitly clear cache - helps prevent memory creep with offloading
        if i % 2 == 0: # More frequent clearing might be needed
             torch.cuda.empty_cache()

    print("Denoising loop finished.")

    # 7. Post-process
    print("Unpacking latents...")
    # Pass the original vae_scale_factor (e.g., 8) for unpacking calculation
    latents = _unpack_latents(latents, height, width, vae_scale_factor_for_latents)
    print(f"  Latents shape (unpacked): {latents.shape}")

    print("Decoding latents...")
    # Ensure latents match VAE dtype before scaling and decoding
    latents = latents.to(vae.dtype)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]
    print("  Decoded image shape:", image.shape)

    print("Postprocessing image...")
    # image_processor expects float32 input typically
    image = image_processor.postprocess(image.to(torch.float32), output_type="pil")

    print("--- Custom Generation Function Finished ---")
    return image

# --- Fixed-Point Inversion Function (Step 2.1) ---
@torch.no_grad()
def invert_image_fixed_point(
    source_image_pil: Image.Image,
    prompt: Optional[str],
    # Accept the full pipeline object
    pipeline: FluxPipeline,
    # Keep device/dtype for tensor creation if needed outside pipeline calls
    device: torch.device,
    dtype: torch.dtype,
    num_inversion_steps: int,
    num_inner_steps: int,
    guidance_scale: float = 0.0,
    max_sequence_length: int = 512,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Performs fixed-point inversion (Stage 1 of Unveil) for a given image and prompt.

    Args:
        source_image_pil: The PIL Image to invert.
        prompt: The text prompt associated with the image.
        pipeline: The loaded FluxPipeline object (potentially with offload enabled).
        device: The target device.
        dtype: The target data type.
        num_inversion_steps (T): Number of outer inversion steps.
        num_inner_steps (I): Number of inner refinement steps.
        guidance_scale: Classifier-free guidance scale.
        max_sequence_length: Max sequence length for tokenizer.

    Returns:
        A tuple containing:
            - x0_latent_packed: The inverted latent code (packed) at step 0 on the device.
            - latent_trajectory: A list of packed latents [x0, x1, ..., xT] on CPU.
    """
    print("--- Starting Fixed-Point Inversion ---")
    # Get components from pipeline object
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer_2 = pipeline.tokenizer_2
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    image_processor = pipeline.image_processor
    # Note: device/dtype are also available via pipeline.device, pipeline.dtype

    batch_size = 1
    num_images_per_prompt = 1

    # 1. Preprocess source image
    print("Preprocessing source image...")
    # Use pipeline's image processor
    image = image_processor.preprocess(source_image_pil)
    # Move image tensor explicitly to target device/dtype (pipeline preprocess might not)
    image = image.to(device=device, dtype=dtype)
    height, width = image.shape[-2:]
    print(f"  Image shape: {image.shape}, Device: {image.device}")

    # 2. Encode source image to target latents
    print("Encoding image to latents...")
    # Use pipeline's VAE - Offload hooks *should* handle device placement
    print(f"  VAE device hint (offloaded?): {vae.device}")
    print(f"  Image tensor device before encode: {image.device}")
    # Ensure image is compatible with VAE's expected dtype if necessary
    x_target_unpacked = vae.encode(image.to(vae.dtype)).latent_dist.sample()
    print("  VAE encoding successful.")
    # Scale and pack latents, ensuring they end up on the target device/dtype
    x_target_unpacked = (x_target_unpacked - vae.config.shift_factor) * vae.config.scaling_factor
    x_target_packed = _pack_latents(x_target_unpacked.to(device=device, dtype=dtype))
    print(f"  Packed target latents shape: {x_target_packed.shape}")

    # 3. Encode prompt
    # Use the custom function, passing necessary pipeline components
    if prompt is None:
        raise NotImplementedError("Unprompted inversion is not yet implemented.")
    print(f"Encoding prompt: '{prompt}'")
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_custom(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        text_encoder=text_encoder, # From pipeline
        tokenizer=tokenizer,       # From pipeline
        text_encoder_2=text_encoder_2, # From pipeline
        tokenizer_2=tokenizer_2,     # From pipeline
        tokenizer_max_length=tokenizer.model_max_length,
    )
    # Ensure embeddings are on the correct device for the transformer
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    text_ids = text_ids.to(device=device, dtype=dtype)
    print(f"  Prompt Embeds shape: {prompt_embeds.shape}, Pooled shape: {pooled_prompt_embeds.shape}")

    # 4. Prepare latent_image_ids
    vae_scale_factor = image_processor.vae_scale_factor
    latent_height_unpacked = height // vae_scale_factor
    latent_width_unpacked = width // vae_scale_factor
    latent_height_for_ids = latent_height_unpacked // 2
    latent_width_for_ids = latent_width_unpacked // 2
    latent_image_ids = _prepare_latent_image_ids(
        batch_size * num_images_per_prompt,
        latent_height_for_ids,
        latent_width_for_ids,
        device, # Use target device
        dtype  # Use target dtype
    )
    print(f"  Latent IDs shape: {latent_image_ids.shape}")

    # 5. Calculate sigmas and set timesteps using pipeline's scheduler
    sigma_min_val = 0.002 # Default for FlowMatchEulerDiscreteScheduler
    print(f"  Using hardcoded sigma_min: {sigma_min_val}")
    sigmas = np.linspace(1.0, sigma_min_val, num_inversion_steps + 1)
    # Pass the numpy array directly, the scheduler handles conversion and device placement
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    print(f"  Set {len(scheduler.timesteps)} timesteps using {len(sigmas)} sigmas.")
    # Check if the number of timesteps is T (num_inversion_steps)
    if len(scheduler.timesteps) != num_inversion_steps:
         print(f"Warning: Scheduler set {len(scheduler.timesteps)} timesteps, expected {num_inversion_steps}.")

    # 6. Prepare guidance embeds
    guidance = None
    if getattr(transformer.config, "guidance_embeds", False):
         guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
         guidance = guidance.expand(batch_size * num_images_per_prompt)
         print("  Guidance embedding enabled for inversion.")

    # 7. Inversion loop
    print(f"Starting inversion loop: T={num_inversion_steps}, I={num_inner_steps}")
    latents = x_target_packed.to(device=device, dtype=dtype)
    latent_trajectory = [latents.detach().cpu()] # Store xT (initial state)

    # scheduler.timesteps should have T+1 elements [t_T, t_{T-1}, ..., t_0]
    # We need to loop T times, calculating x_{T-1} down to x_0.
    # Loop over timesteps t_T down to t_1. This involves T steps.
    timesteps_to_iterate = scheduler.timesteps[:-1] # Exclude t_0
    print(f"  Iterating over {len(timesteps_to_iterate)} timesteps (T={num_inversion_steps}).")

    for i, t_current in tqdm(enumerate(timesteps_to_iterate), desc="Fixed-Point Inversion", total=len(timesteps_to_iterate)):
        # `t_current` is the timestep at the *start* of the interval (e.g., t_T, t_{T-1}, ..., t_1)
        # We need sigmas corresponding to the interval [t_prev, t_current]
        # Get the index corresponding to t_current in the *original* sigmas array [sigma_0, ..., sigma_T]
        # `scheduler.timesteps` is reversed sigmas (excluding sigma_0). timesteps = [sigma_T, ..., sigma_1]
        # `timesteps_to_iterate` = [sigma_T, ..., sigma_1]
        # `i` goes from 0 to T-1.
        # `t_current` = sigma_{T-i}
        # We need interval [sigma_{T-i-1}, sigma_{T-i}]
        idx = num_inversion_steps - i # Index in original sigmas array: T, T-1, ..., 1

        sigma_idx = idx
        sigma_t        = scheduler.sigmas[sigma_idx - 1] # sigma_{T-1}, ..., sigma_0
        sigma_t_plus_1 = scheduler.sigmas[sigma_idx]     # sigma_T, ..., sigma_1

        # Time for model: Use sigma_t (time level being estimated)
        time_t = sigma_t.expand(batch_size).to(device=device, dtype=latents.dtype)

        x_t_plus_1 = latents # State from previous step (starts as x_T)
        x_t_accum = torch.zeros_like(latents, device=device)
        x_t_estim_inner = x_t_plus_1.clone()

        # Inner loop I
        for _ in range(num_inner_steps):
            # Call pipeline's transformer - Offload hooks *should* handle device placement
            velocity = transformer(
                hidden_states=x_t_estim_inner.to(transformer.dtype),
                timestep=(time_t / 1000).to(transformer.dtype),
                guidance=guidance.to(transformer.dtype) if guidance is not None else None,
                pooled_projections=pooled_prompt_embeds.to(transformer.dtype),
                encoder_hidden_states=prompt_embeds.to(transformer.dtype),
                txt_ids=text_ids.to(transformer.dtype),
                img_ids=latent_image_ids.to(transformer.dtype),
                return_dict=False,
            )[0]
            # Velocity should be on the correct device after transformer call (if offload worked)
            velocity = velocity.to(device=device)

            dt_for_eq9 = sigma_t - sigma_t_plus_1
            x_t_estim_eq9 = x_t_plus_1 + dt_for_eq9 * velocity
            x_t_accum += x_t_estim_eq9

            x_t_estim_inner = x_t_plus_1 + dt_for_eq9 * velocity

        x_t_final = x_t_accum / num_inner_steps
        latents = x_t_final
        latent_trajectory.append(latents.detach().cpu())

    print("Inversion loop finished.")
    latent_trajectory.reverse() # Should be [x0, x1, ..., xT]
    x0_latent_packed = latents # Should be x0

    print(f"  Final x0 latent shape: {x0_latent_packed.shape}")
    print(f"  Trajectory length: {len(latent_trajectory)}")
    print("--- Fixed-Point Inversion Finished ---")

    return x0_latent_packed, latent_trajectory


# --- Script Entry Point ---
if __name__ == "__main__":
    print(f"--- Script Start ---")
    # Define device and dtype explicitly at the start
    model_id = "black-forest-labs/FLUX.1-schnell"
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using model: {model_id}")
    print(f"Using dtype: {dtype}")
    print(f"Using device: {device}")

    # Create output dirs
    output_dir_inv = "output_custom_pipeline_inversion"
    output_dir_recon = "output_custom_pipeline_reconstruction"
    output_dir_edit = "output_custom_pipeline_editing"
    os.makedirs(output_dir_inv, exist_ok=True)
    os.makedirs(output_dir_recon, exist_ok=True)
    os.makedirs(output_dir_edit, exist_ok=True)
    output_dir_gen = "output_custom_pipeline"
    os.makedirs(output_dir_gen, exist_ok=True)

    # Load the original pipeline
    print("Loading base pipeline...")
    t_load_start = time.time()
    base_pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    t_load_end = time.time()
    print(f"Base pipeline loaded in {t_load_end - t_load_start:.2f} seconds.")

    # Enable sequential CPU offload BEFORE extracting/using components
    print("Enabling sequential CPU offload on base pipeline...")
    if device.type == 'cuda':
        # Note: Offload hooks might not work perfectly when calling components directly later
        base_pipe.enable_sequential_cpu_offload(device=device)
        print("Sequential CPU offload enabled.")
    else:
        print("Running on CPU, moving full pipeline to CPU.")
        base_pipe.to(device)

    # Extract components - We might not need these if we pass base_pipe
    # print("Extracting components...")
    # vae = base_pipe.vae
    # text_encoder = base_pipe.text_encoder
    # ... etc ...
    # print(f"  VAE device after extraction: {vae.device}")

    # --- Verification for Step 2.1 ---
    print("\n--- Running Verification for Step 2.1 ---")
    # Test parameters
    test_image_path = "test_images/test_image_portrait.jpeg"
    # Read prompt from corresponding .txt file if it exists
    prompt_file_path = os.path.splitext(test_image_path)[0] + ".txt"
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, 'r') as f:
            test_prompt = f.read().strip()
        print(f"Loaded prompt from {prompt_file_path}: '{test_prompt}'")
    else:
        # Fallback prompt if file doesn't exist
        test_prompt = "A photograph of a man with long hair in his 30s, soft smile" # From plan Step 3.1
        print(f"Prompt file {prompt_file_path} not found. Using fallback prompt: '{test_prompt}'")

    num_inversion_steps_test = 30 # T
    num_inner_steps_test = 3    # I
    guidance_scale_test = 0.0
    max_sequence_length_test = 512 # Default

    print(f"Loading test image: {test_image_path}")
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at {test_image_path}. Please place test images in a 'test_images' directory.")
        # Handle error appropriately - maybe skip this test? For now, exit.
        import sys
        sys.exit(f"Test image not found: {test_image_path}")


    source_image = Image.open(test_image_path).convert("RGB")
    # Optional: Resize image if needed, Flux works well with 1024x1024
    # source_image = source_image.resize((1024, 1024))
    print(f"Image size: {source_image.size}")

    # Get the scheduler from the pipeline *before* calling the inversion function
    # to check its state after timesteps are set inside the function.
    verification_scheduler = base_pipe.scheduler

    print(f"Running invert_image_fixed_point with T={num_inversion_steps_test}, I={num_inner_steps_test}...")
    t_inv_start = time.time()
    try:
        x0_latent, trajectory = invert_image_fixed_point(
            source_image_pil=source_image,
            prompt=test_prompt,
            # Pass pipeline object
            pipeline=base_pipe,
            # Keep device/dtype for tensor creation
            device=device,
            dtype=dtype,
            num_inversion_steps=num_inversion_steps_test,
            num_inner_steps=num_inner_steps_test,
            guidance_scale=guidance_scale_test,
            max_sequence_length=max_sequence_length_test
        )
        t_inv_end = time.time()
        print(f"Inversion finished in {t_inv_end - t_inv_start:.2f} seconds.")

        print("\nVerification Results:")
        print(f"  - Returned x0_latent shape: {x0_latent.shape}")
        print(f"  - Returned x0_latent dtype: {x0_latent.dtype}")
        print(f"  - Returned x0_latent device: {x0_latent.device}")
        print(f"  - Returned trajectory length: {len(trajectory)}")
        if trajectory:
            print(f"  - Trajectory[0] (x0) shape: {trajectory[0].shape}")
            print(f"  - Trajectory[0] (x0) dtype: {trajectory[0].dtype}")
            print(f"  - Trajectory[0] (x0) device: {trajectory[0].device}")
            print(f"  - Trajectory[-1] (xT) shape: {trajectory[-1].shape}")
            print(f"  - Trajectory[-1] (xT) dtype: {trajectory[-1].dtype}")
            print(f"  - Trajectory[-1] (xT) device: {trajectory[-1].device}")

        # Basic checks
        # The loop now runs T times (num_inversion_steps).
        # The trajectory stores the initial xT + T results = T+1 elements.
        expected_traj_len = num_inversion_steps_test + 1
        print(f"  (Debug: Loop should run {num_inversion_steps_test} times)")

        if len(trajectory) == expected_traj_len:
            print(f"  - Trajectory length check: PASSED ({len(trajectory)} == {expected_traj_len})")
        else:
            print(f"  - Trajectory length check: FAILED (Expected {expected_traj_len}, Got {len(trajectory)})")

        # Check if x0 matches trajectory[0]
        if trajectory and torch.allclose(x0_latent.cpu(), trajectory[0], atol=1e-5):
             print(f"  - x0 vs trajectory[0] check: PASSED")
        elif not trajectory:
             print(f"  - x0 vs trajectory[0] check: SKIPPED (empty trajectory)")
        else:
             # Add more detailed comparison if fails
             diff = torch.abs(x0_latent.cpu() - trajectory[0]).max()
             print(f"  - x0 vs trajectory[0] check: FAILED (Max difference: {diff.item()})")

        print("--- Step 2.1 Verification Complete ---")

    except NotImplementedError as e:
        print(f"\nERROR: {e}")
        print("Step 2.1 verification skipped due to missing feature.")
    except Exception as e:
        print(f"\n--- ERROR during Step 2.1 verification ---")
        import traceback
        traceback.print_exc()
        print(f"Error details: {e}")
        if "out of memory" in str(e).lower():
             print("CUDA Out of Memory detected during inversion.")
             # Add suggestions specific to inversion maybe?
             print("- Ensure 'enable_sequential_cpu_offload' was successfully called.")
             print("- Try reducing num_inner_steps (I).")
             print("- Verify system RAM is sufficient.")

    # --- Optional: Keep original generation call for comparison/testing ---
    # print("\n--- Running Original Custom Generation (Optional) ---")
    # ... (keep the original call to run_custom_generation if desired) ...

    print("\n--- Script End ---") 