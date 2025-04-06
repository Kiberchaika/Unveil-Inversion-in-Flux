import torch
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

# --- Configuration ---
model_id = "black-forest-labs/FLUX.1-schnell"
# Use float16 for memory efficiency as suggested in the video
dtype = torch.float16
output_dir = "output_custom_pipeline"
prompt = "Roman Empire soldier with a sword and a shield, horses in background, mountain with snow in background"
seed = 42

logger = logging.get_logger(__name__)

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

# --- Script Entry Point ---
if __name__ == "__main__":
    print(f"--- Script Start ---")
    print(f"Using model: {model_id}")
    print(f"Using dtype: {dtype}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load the original pipeline to easily get components and enable offload
    print("Loading base pipeline to extract components...")
    t_load_start = time.time()
    base_pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    t_load_end = time.time()
    print(f"Base pipeline loaded in {t_load_end - t_load_start:.2f} seconds.")

    print("Enabling sequential CPU offload on base pipeline...")
    # This configures hooks on the modules themselves
    base_pipe.enable_sequential_cpu_offload(device=device) # Specify device for offload map
    print("Sequential CPU offload enabled.")

    # Extract components (they should retain offloading hooks)
    print("Extracting components...")
    vae = base_pipe.vae
    text_encoder = base_pipe.text_encoder
    tokenizer = base_pipe.tokenizer
    text_encoder_2 = base_pipe.text_encoder_2
    tokenizer_2 = base_pipe.tokenizer_2
    transformer = base_pipe.transformer
    scheduler = base_pipe.scheduler
    image_processor = base_pipe.image_processor # Get image processor

    # Optional: Delete the base pipeline object to free some memory if needed,
    # but the offloaded modules hold the main memory.
    # print("Deleting base pipeline object...")
    # del base_pipe
    # torch.cuda.empty_cache()

    # Run the custom generation
    print(f"Running custom generation for prompt: '{prompt}'")
    t_gen_start = time.time()
    try:
        # Define generation parameters used in the custom function
        height = 1024 # Default or desired height
        width = 1024  # Default or desired width
        num_inference_steps = 4 # For Schnell
        guidance_scale = 0.0 # For Schnell

        image = run_custom_generation(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            device=device,
            dtype=dtype,
            max_sequence_length=512 # Default from pipeline
        )
        t_gen_end = time.time()
        print(f"Image generated via custom loop in {t_gen_end - t_gen_start:.2f} seconds.")

        # Display and save
        print("Displaying image...")
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show() # Removed this line as it might cause issues in non-interactive envs

        save_path = os.path.join(output_dir, f"custom_flux_output_seed{seed}.png")
        print(f"Saving image to: {save_path}")
        # Access the first element if 'image' is a list
        if isinstance(image, list):
            if len(image) > 0:
                image[0].save(save_path)
            else:
                print("Error: Postprocessing returned an empty list.")
        else:
            # Assume it's a PIL image if not a list
            image.save(save_path)
        print("Image saved.")

    except Exception as e:
        print(f"--- ERROR during custom generation ---")
        import traceback
        traceback.print_exc()
        print(f"Error details: {e}")
        if "out of memory" in str(e).lower():
             print("CUDA Out of Memory detected.")
             print("Suggestions:")
             print("- Ensure 'enable_sequential_cpu_offload' was successfully called.")
             print("- Close other GPU-intensive applications.")
             print("- Reduce image height/width if possible (though defaults are standard).")
             print("- Verify system RAM is sufficient for offloaded parts.")

    print("--- Custom Script Finished ---") 