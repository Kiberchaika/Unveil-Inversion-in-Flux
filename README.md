# Flux Image Editing (Based on Inversion/Invariance Paper)

This project is an implementation of the text-guided image editing techniques proposed in the paper **"Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing"** (Xu et al.).

It utilizes the **`black-forest-labs/FLUX.1-schnell`** model via the **Hugging Face Diffusers library** as the base generative model.

The core goal is to adapt the `diffusers` Flux pipeline to incorporate:

1.  **Two-Stage Flow Inversion:** A method for faithfully inverting a real image back to its initial noise latent (`x0`) and trajectory, aiming to stay close to the model's generative manifold for better editability (Sec 4.2 of the paper).
2.  **AdaLN-Based Invariance Control:** A technique leveraging Adaptive Layer Normalization (AdaLN) feature swapping based on source/target prompt differences to control which parts of the original image are preserved during editing (Sec 5 of the paper).

This fork focuses specifically on implementing and testing these paper-specific algorithms for versatile image editing within the Flux framework.

## Development Plan

The detailed implementation steps and verification strategy are outlined in [unveil_inversion_plan.md](./unveil_inversion_plan.md).

*(Original README content, if any, can be added below or integrated as needed)*
