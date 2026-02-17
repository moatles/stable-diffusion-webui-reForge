from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from blendmodes.blend import BlendType, blendLayers
from einops import rearrange, repeat
from PIL import Image, ImageOps
from skimage import exposure

import modules.face_restoration
import modules.images as images
import modules.paths as paths
import modules.processing_scripts.comments as comments_parser
import modules.sd_hijack
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
import modules.shared as shared
import modules.styles
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
from ldm_patched.modules.model_sampling import rescale_zero_terminal_snr_sigmas
from modules import (
    devices,
    errors,
    extra_networks,
    infotext_utils,
    lowvram,
    masking,
    profiling,
    prompt_parser,
    rng,
    scripts,
    sd_samplers,
    sd_samplers_common,
    sd_unet,
    sd_vae_approx,
)
from modules.rng import slerp
from modules.sd_hijack import model_hijack
from modules.sd_models import apply_token_merging
from modules.sd_samplers_common import (
    approximation_indexes,
    decode_first_stage,
    images_tensor_to_samples,
)
from modules.shared import cmd_opts, opts, state
from modules_forge.forge_util import apply_circular_forge

# =============================================================================
# Constants
# =============================================================================

# Latent space channel count (VAE compression)
LATENT_CHANNELS = 4

# VAE downscaling factor (512px image → 64px latent)
VAE_SCALE_FACTOR = 8

# Maximum random seed value
MAX_SEED_VALUE = 4294967294

# CLIP skip bounds
MIN_CLIP_SKIP = 2
MAX_CLIP_SKIP = 12

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def temporary_clip_skip(clip_skip: int):
    """Temporarily change CLIP skip setting."""
    original = opts.CLIP_stop_at_last_layers
    opts.CLIP_stop_at_last_layers = clip_skip
    try:
        yield
    finally:
        opts.CLIP_stop_at_last_layers = original


@contextmanager
def temporary_options(**settings):
    """Temporarily change multiple options."""
    originals = {}
    for key, value in settings.items():
        if hasattr(opts, key):
            originals[key] = getattr(opts, key)
            setattr(opts, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(opts, key, value)


# =============================================================================
# Utility Functions
# =============================================================================

def setup_color_correction(image: Image.Image) -> np.ndarray:
    """Calibrate color correction from reference image.

    Args:
        image: Reference image for color calibration.

    Returns:
        LAB color space representation for histogram matching.
    """
    logger.debug("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction: np.ndarray,
                           original_image: Image.Image) -> Image.Image:
    """Apply color correction to match reference histogram.

    Args:
        correction: LAB color space reference from setup_color_correction.
        original_image: Image to apply correction to.

    Returns:
        Color-corrected image.
    """
    logger.debug("Applying color correction.")

    # Convert to LAB color space for histogram matching
    original_lab = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2LAB)

    # Match histograms
    matched_lab = exposure.match_histograms(original_lab, correction, channel_axis=2)

    # Convert back to RGB
    matched_rgb = cv2.cvtColor(matched_lab.astype("uint8"), cv2.COLOR_LAB2RGB)

    # Blend with original luminosity
    image = Image.fromarray(matched_rgb)
    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image.convert('RGB')


def uncrop(image: Image.Image, dest_size: Tuple[int, int],
           paste_loc: Tuple[int, int, int, int]) -> Image.Image:
    """Place image onto a larger canvas at specified location.

    Args:
        image: Image to place.
        dest_size: Size of destination canvas (width, height).
        paste_loc: Location tuple (x, y, width, height).

    Returns:
        New RGBA image with original placed at location.
    """
    x, y, w, h = paste_loc
    base_image = Image.new('RGBA', dest_size)
    image = images.resize_image(1, image, w, h)
    base_image.paste(image, (x, y))
    return base_image


def apply_overlay(
    image: Image.Image,
    paste_loc: Optional[Tuple[int, int, int, int]],
    overlay: Optional[Image.Image]
) -> Tuple[Image.Image, Image.Image]:
    """Apply overlay image with optional uncropping.

    Args:
        image: Base image.
        paste_loc: Optional paste location for uncropping.
        overlay: Optional overlay image.

    Returns:
        Tuple of (composited image, original denoised image copy).
    """
    if overlay is None:
        return image, image.copy()

    if paste_loc is not None:
        image = uncrop(image, (overlay.width, overlay.height), paste_loc)

    original_denoised_image = image.copy()

    # Composite overlay if it has alpha channel
    if overlay.mode == 'RGBA':
        image = image.convert('RGBA')
        image.alpha_composite(overlay)
        image = image.convert('RGB')
    else:
        image.paste(overlay, (0, 0))

    return image, original_denoised_image


def create_binary_mask(image: Image.Image, round_mask: bool = True) -> Image.Image:
    """Create binary mask from image, handling alpha channels.

    Args:
        image: Input image (may have alpha channel).
        round_mask: Whether to round mask values to 0 or 255.

    Returns:
        Grayscale mask image.
    """
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        alpha = image.split()[-1].convert("L")
        if round_mask:
            alpha = alpha.point(lambda x: 255 if x > 128 else 0)
        return alpha
    return image.convert('L')


def txt2img_image_conditioning(
    sd_model: Any,
    x: torch.Tensor,
    width: int,
    height: int
) -> torch.Tensor:
    """Generate conditioning for txt2img based on model type.

    Args:
        sd_model: Stable Diffusion model.
        x: Input tensor for shape reference.
        width: Image width.
        height: Image height.

    Returns:
        Image conditioning tensor appropriate for model type.
    """
    conditioning_key = sd_model.model.conditioning_key

    if conditioning_key in {'hybrid', 'concat'}:  # Inpainting models
        # Create masked image with all 0.5 (full mask)
        image_conditioning = torch.ones(
            x.shape[0], 3, height, width, device=x.device
        ) * 0.5
        image_conditioning = images_tensor_to_samples(
            image_conditioning,
            approximation_indexes.get(opts.sd_vae_encode_method)
        )

        # Add fake mask (all 1s)
        image_conditioning = torch.nn.functional.pad(
            image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0
        )
        return image_conditioning.to(x.dtype)

    elif conditioning_key == "crossattn-adm":  # UnCLIP models
        return x.new_zeros(
            x.shape[0],
            2 * sd_model.noise_augmentor.time_embed.dim,
            dtype=x.dtype,
            device=x.device
        )

    else:
        # SDXL inpainting special case
        if getattr(sd_model, 'is_sdxl_inpaint', False):
            image_conditioning = torch.ones(
                x.shape[0], 3, height, width, device=x.device
            ) * 0.5
            image_conditioning = images_tensor_to_samples(
                image_conditioning,
                approximation_indexes.get(opts.sd_vae_encode_method)
            )
            image_conditioning = torch.nn.functional.pad(
                image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0
            )
            return image_conditioning.to(x.dtype)

        # Dummy conditioning for non-inpainting models
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


def get_fixed_seed(seed: Union[int, str, None]) -> int:
    """Convert seed to fixed integer.

    Args:
        seed: Input seed (int, string, None, or -1 for random).

    Returns:
        Fixed integer seed.
    """
    if seed in (None, '', -1):
        return int(random.randrange(MAX_SEED_VALUE))

    if isinstance(seed, str):
        try:
            return int(seed)
        except ValueError:
            return int(random.randrange(MAX_SEED_VALUE))

    return int(seed)


def fix_seed(p: 'StableDiffusionProcessing'):
    """Fix seeds in processing object.

    Args:
        p: Processing object to fix seeds for.
    """
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def program_version() -> Optional[str]:
    """Get program version from git.

    Returns:
        Version string or None if unavailable.
    """
    import launch
    res = launch.git_tag()
    return None if res == "<none>" else res


def old_hires_fix_first_pass_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Calculate first pass dimensions using old hires fix algorithm.

    Args:
        width: Target width.
        height: Target height.

    Returns:
        Tuple of (first_pass_width, first_pass_height).
    """
    desired_pixel_count = 512 * 512
    actual_pixel_count = width * height
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    width = math.ceil(scale * width / 64) * 64
    height = math.ceil(scale * height / 64) * 64
    return width, height


# =============================================================================
# Main Processing Classes
# =============================================================================

@dataclass(repr=False)
class StableDiffusionProcessing:
    """Base class for Stable Diffusion processing."""

    # Input parameters - sd_model kept for API compatibility
    sd_model: Any = None
    outpath_samples: Optional[str] = None
    outpath_grids: Optional[str] = None
    prompt: Union[str, List[str]] = ""
    prompt_for_display: Optional[str] = None
    negative_prompt: Union[str, List[str]] = ""
    styles: Optional[List[str]] = None
    seed: Union[int, List[int]] = -1
    subseed: Union[int, List[int]] = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    seed_enable_extras: bool = True
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    restore_faces: Optional[bool] = None
    tiling: Optional[bool] = None
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    extra_generation_params: Optional[Dict[str, Any]] = None
    overlay_images: Optional[List[Image.Image]] = None
    eta: Optional[float] = None
    do_not_reload_embeddings: bool = False
    denoising_strength: Optional[float] = None
    ddim_discretize: Optional[str] = None
    s_min_uncond: Optional[float] = None
    s_churn: Optional[float] = None
    s_tmax: Optional[float] = None
    s_tmin: Optional[float] = None
    s_noise: Optional[float] = None
    override_settings: Optional[Dict[str, Any]] = None
    override_settings_restore_afterwards: bool = True
    sampler_index: Optional[int] = None
    refiner_checkpoint: Optional[str] = None
    refiner_switch_at: Optional[float] = None
    token_merging_ratio: float = 0
    token_merging_ratio_hr: float = 0
    disable_extra_networks: bool = False
    firstpass_image: Optional[Image.Image] = None

    # Script-related fields
    scripts_value: Optional[scripts.ScriptRunner] = field(default=None, init=False)
    script_args_value: Optional[List] = field(default=None, init=False)
    scripts_setup_complete: bool = field(default=False, init=False)

    # Cached conditionings - class level for persistence
    # Format: [params, conditioning, extra_params]
    cached_uc: List = field(default_factory=lambda: [None, None, None], init=False)
    cached_c: List = field(default_factory=lambda: [None, None, None], init=False)

    # Processing state
    comments: Dict[str, int] = field(default_factory=dict, init=False)
    sampler: Optional[sd_samplers_common.Sampler] = field(default=None, init=False)
    is_using_inpainting_conditioning: bool = field(default=False, init=False)
    paste_to: Optional[Tuple[int, int, int, int]] = field(default=None, init=False)
    is_hr_pass: bool = field(default=False, init=False)
    c: Optional[Any] = field(default=None, init=False)
    uc: Optional[Any] = field(default=None, init=False)
    rng: Optional[rng.ImageRNG] = field(default=None, init=False)
    step_multiplier: int = field(default=1, init=False)
    color_corrections: Optional[List[np.ndarray]] = field(default=None, init=False)
    clip_skip: int = field(default=1, init=False)
    firstpass_steps: int = field(default=0, init=False)

    # Batch processing
    all_prompts: Optional[List[str]] = field(default=None, init=False)
    all_negative_prompts: Optional[List[str]] = field(default=None, init=False)
    all_seeds: Optional[List[int]] = field(default=None, init=False)
    all_subseeds: Optional[List[int]] = field(default=None, init=False)
    iteration: int = field(default=0, init=False)
    main_prompt: Optional[str] = field(default=None, init=False)
    main_negative_prompt: Optional[str] = field(default=None, init=False)

    # Current batch
    prompts: Optional[List[str]] = field(default=None, init=False)
    negative_prompts: Optional[List[str]] = field(default=None, init=False)
    seeds: Optional[List[int]] = field(default=None, init=False)
    subseeds: Optional[List[int]] = field(default=None, init=False)
    extra_network_data: Optional[Dict] = field(default=None, init=False)

    # Metadata
    user: Optional[str] = field(default=None, init=False)
    sd_model_name: Optional[str] = field(default=None, init=False)
    sd_model_hash: Optional[str] = field(default=None, init=False)
    sd_vae_name: Optional[str] = field(default=None, init=False)
    sd_vae_hash: Optional[str] = field(default=None, init=False)

    is_api: bool = field(default=False, init=False)

    # Storage for intermediate results
    latents_after_sampling: List[torch.Tensor] = field(default_factory=list, init=False)
    pixels_after_sampling: List[Image.Image] = field(default_factory=list, init=False)

    # Dynamic clip skip
    dynamic_clip_skip_schedule: Optional[List[int]] = field(default=None, init=False)
    dynamic_clip_skip_sets: Optional[Dict[int, Tuple]] = field(default=None, init=False)

    # Noise modification
    modified_noise: Optional[torch.Tensor] = field(default=None, init=False)
    extra_result_images: List[Image.Image] = field(default_factory=list, init=False)

    # Sampler override
    sampler_noise_scheduler_override: Optional[Any] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize processing parameters."""
        if self.sampler_index is not None:
            print(
                "Warning: sampler_index argument for StableDiffusionProcessing "
                "does nothing; use sampler_name",
                file=sys.stderr
            )

        self.refiner_checkpoint_info = None

        if self.styles is None:
            self.styles = []

        self.extra_generation_params = self.extra_generation_params or {}
        self.override_settings = self.override_settings or {}

        if not self.seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

    def fill_fields_from_opts(self):
        """Fill optional parameters from global options."""
        if opts.sd_sampling == "A1111":
            self.s_min_uncond = self.s_min_uncond or opts.s_min_uncond
            self.s_churn = self.s_churn or opts.s_churn
            self.s_tmin = self.s_tmin or opts.s_tmin
            self.s_tmax = (self.s_tmax or opts.s_tmax) or float('inf')
            self.s_noise = self.s_noise or opts.s_noise
        else:
            # Set defaults for non-A1111 sampling
            if self.s_min_uncond is None:
                self.s_min_uncond = 0.0
            if self.s_churn is None:
                self.s_churn = 0.0
            if self.s_tmin is None:
                self.s_tmin = 0.0
            if self.s_tmax is None:
                self.s_tmax = float('inf')
            if self.s_noise is None:
                self.s_noise = 1.0

    def get_sd_model(self) -> Any:
        """Get current SD model, falling back to shared model if not set."""
        if self.sd_model is not None:
            return self.sd_model
        return shared.sd_model

    @property
    def scripts(self) -> Optional[scripts.ScriptRunner]:
        """Get scripts runner."""
        return self.scripts_value

    @scripts.setter
    def scripts(self, value: Optional[scripts.ScriptRunner]):
        """Set scripts runner and setup if ready."""
        self.scripts_value = value
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self._setup_scripts()

    @property
    def script_args(self) -> Optional[List]:
        """Get script arguments."""
        return self.script_args_value

    @script_args.setter
    def script_args(self, value: Optional[List]):
        """Set script arguments and setup if ready."""
        self.script_args_value = value
        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self._setup_scripts()

    def _setup_scripts(self):
        """Initialize scripts."""
        self.scripts_setup_complete = True
        self.scripts.setup_scrips(self, is_ui=not self.is_api)

    def comment(self, text: str):
        """Add comment to processing metadata."""
        self.comments[text] = 1

    def txt2img_image_conditioning(
        self,
        x: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> torch.Tensor:
        """Wrapper for txt2img image conditioning."""
        model = self.get_sd_model()
        self.is_using_inpainting_conditioning = (
            model.model.conditioning_key in {'hybrid', 'concat'}
        )
        return txt2img_image_conditioning(
            model, x, width or self.width, height or self.height
        )

    def depth2img_image_conditioning(self, source_image: torch.Tensor) -> torch.Tensor:
        """Generate depth-based conditioning."""
        model = self.get_sd_model()
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(
            transformed["midas_in"][None, ...]
        ).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = images_tensor_to_samples(
            source_image * 0.5 + 0.5,
            approximation_indexes.get(opts.sd_vae_encode_method)
        )
        conditioning = torch.nn.functional.interpolate(
            model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        depth_min, depth_max = torch.aminmax(conditioning)
        conditioning = 2.0 * (conditioning - depth_min) / (depth_max - depth_min) - 1.0
        return conditioning

    def edit_image_conditioning(self, source_image: torch.Tensor) -> torch.Tensor:
        """Generate edit-based conditioning."""
        return shared.sd_model.encode_first_stage(source_image).mode()

    def unclip_image_conditioning(self, source_image: torch.Tensor) -> torch.Tensor:
        """Generate unclip conditioning."""
        model = self.get_sd_model()
        c_adm = model.embedder(source_image)
        if model.noise_augmentor is not None:
            noise_level = 0
            noise_level_tensor = repeat(
                torch.tensor([noise_level]).to(c_adm.device),
                '1 -> b', b=c_adm.shape[0]
            )
            c_adm, noise_level_emb = model.noise_augmentor(
                c_adm, noise_level=noise_level_tensor
            )
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(
        self,
        source_image: torch.Tensor,
        latent_image: torch.Tensor,
        image_mask: Optional[Image.Image] = None,
        round_image_mask: bool = True
    ) -> torch.Tensor:
        """Generate inpainting conditioning."""
        model = self.get_sd_model()
        self.is_using_inpainting_conditioning = True

        # Handle mask
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

                if round_image_mask:
                    conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])

        # Apply mask weight
        conditioning_mask = conditioning_mask.to(
            device=source_image.device, dtype=source_image.dtype
        )
        mask_weight = getattr(
            self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight
        )
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            mask_weight
        )

        # Encode masked image
        conditioning_image = model.get_first_stage_encoding(
            model.encode_first_stage(conditioning_image)
        )

        # Prepare conditioning tensor
        conditioning_mask = torch.nn.functional.interpolate(
            conditioning_mask, size=latent_image.shape[-2:]
        )
        conditioning_mask = conditioning_mask.expand(
            conditioning_image.shape[0], -1, -1, -1
        )
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)

        return image_conditioning.to(shared.device).type(model.dtype)

    def img2img_image_conditioning(
        self,
        source_image: torch.Tensor,
        latent_image: torch.Tensor,
        image_mask: Optional[Image.Image] = None,
        round_image_mask: bool = True
    ) -> torch.Tensor:
        """Generate img2img conditioning based on model type."""
        model = self.get_sd_model()
        source_image = devices.cond_cast_float(source_image)

        # Depth2Image models
        if isinstance(model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)

        # Edit models
        if model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)

        # Inpainting models
        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(
                source_image, latent_image, image_mask, round_image_mask
            )

        # UnCLIP models
        if self.sampler.conditioning_key == "crossattn-adm":
            return self.unclip_image_conditioning(source_image)

        # SDXL inpainting
        if self.sampler.model_wrap.inner_model.is_sdxl_inpaint:
            return self.inpainting_image_conditioning(
                source_image, latent_image, image_mask
            )

        # Default: zero conditioning
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts: List[str], all_seeds: List[int],
             all_subseeds: List[int]):
        """Initialize processing (to be overridden by subclasses)."""
        pass

    def sample(
        self,
        conditioning: Any,
        unconditional_conditioning: Any,
        seeds: List[int],
        subseeds: List[int],
        subseed_strength: float,
        prompts: List[str]
    ) -> Optional[torch.Tensor]:
        """Perform sampling (to be implemented by subclasses)."""
        raise NotImplementedError()

    def close(self):
        """Clean up resources."""
        self.sampler = None
        self.c = None
        self.uc = None
        self.latents_after_sampling.clear()
        self.pixels_after_sampling.clear()
        self.extra_result_images.clear()

        if not opts.persistent_cond_cache:
            StableDiffusionProcessing.cached_c = [None, None, None]
            StableDiffusionProcessing.cached_uc = [None, None, None]

    def get_token_merging_ratio(self, for_hr: bool = False) -> float:
        """Get token merging ratio for current pass."""
        if for_hr:
            return (
                self.token_merging_ratio_hr or
                opts.token_merging_ratio_hr or
                self.token_merging_ratio or
                opts.token_merging_ratio
            )
        return self.token_merging_ratio or opts.token_merging_ratio

    def setup_prompts(self):
        """Setup prompt lists for batch processing."""
        # Handle prompts
        if isinstance(self.prompt, list):
            self.all_prompts = self.prompt
        else:
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]

        # Handle negative prompts
        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)

        # Validate lengths
        if len(self.all_prompts) != len(self.all_negative_prompts):
            raise RuntimeError(
                f"Received different number of prompts ({len(self.all_prompts)}) "
                f"and negative prompts ({len(self.all_negative_prompts)})"
            )

        # Apply styles
        self.all_prompts = [
            shared.prompt_styles.apply_styles_to_prompt(x, self.styles)
            for x in self.all_prompts
        ]
        self.all_negative_prompts = [
            shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles)
            for x in self.all_negative_prompts
        ]

        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    def cached_params(
        self,
        required_prompts: Any,
        steps: int,
        extra_network_data: Dict,
        hires_steps: Optional[int] = None,
        use_old_scheduling: bool = False
    ) -> tuple:
        """Generate cache key for conditioning."""
        return (
            required_prompts,
            steps,
            hires_steps,
            use_old_scheduling,
            opts.CLIP_stop_at_last_layers,
            opts.dynamic_clip_skip_enabled,
            opts.dynamic_clip_skip_start,
            opts.dynamic_clip_skip_use_schedule,
            opts.dynamic_clip_skip_schedule_values,
            shared.sd_model.sd_checkpoint_info,
            extra_network_data,
            opts.sdxl_crop_left,
            opts.sdxl_crop_top,
            self.width,
            self.height,
            opts.fp8_storage,
            opts.cache_fp16_weight,
            opts.emphasis,
        )

    def _get_minimal_clip_skip(self) -> int:
        """Get minimal clip skip value for current model."""
        model = self.get_sd_model()
        cond_model = getattr(model, "cond_stage_model", None) if model else None
        return getattr(cond_model, "minimal_clip_skip", 1) if cond_model else 1

    def _build_dynamic_clip_skip_schedule(
        self,
        total_steps: int,
        start: int
    ) -> List[int]:
        """Build linear clip skip schedule."""
        schedule = []
        current = int(start)
        for _ in range(total_steps):
            schedule.append(current)
            if current > MIN_CLIP_SKIP:
                current -= 1
            if current < MIN_CLIP_SKIP:
                current = MIN_CLIP_SKIP
        return schedule

    def _parse_dynamic_schedule(
        self,
        total_steps: int,
        minimal_clip_skip: int
    ) -> Optional[List[int]]:
        """Parse custom clip skip schedule."""
        raw = opts.dynamic_clip_skip_schedule_values or ""
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        values = []

        for p in parts:
            try:
                v = int(float(p))
                v = max(v, MIN_CLIP_SKIP, minimal_clip_skip)
                v = min(v, MAX_CLIP_SKIP)
                values.append(v)
            except (ValueError, TypeError):
                continue

        if not values:
            return None

        # Extend schedule to total_steps
        schedule = []
        for i in range(total_steps):
            idx = min(i, len(values) - 1)
            schedule.append(values[idx])
        return schedule

    def get_conds_with_caching(
        self,
        function: Callable,
        required_prompts: Any,
        steps: int,
        caches: List[List],
        extra_network_data: Dict,
        hires_steps: Optional[int] = None
    ) -> Any:
        """Get conditioning with caching.

        Args:
            function: Function to compute conditioning.
            required_prompts: Prompts to condition on.
            steps: Number of steps.
            caches: List of cache lists, each [params, conditioning, extra_params].
            extra_network_data: Extra network data.
            hires_steps: Optional hires steps count.

        Returns:
            Computed or cached conditioning.
        """
        # Check old vs new scheduling
        if shared.opts.use_old_scheduling:
            old_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(
                required_prompts, steps, hires_steps, False
            )
            new_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(
                required_prompts, steps, hires_steps, True
            )
            if old_schedules != new_schedules:
                self.extra_generation_params["Old prompt editing timelines"] = True

        # Build cache key
        cache_params = self.cached_params(
            required_prompts, steps, extra_network_data,
            hires_steps, shared.opts.use_old_scheduling
        )

        # Check each cache
        for cache in caches:
            if cache[0] is not None and cache_params == cache[0]:
                if len(cache) > 2 and cache[2] is not None:
                    modules.sd_hijack.model_hijack.extra_generation_params.update(cache[2])
                return cache[1]

        # Compute and cache in first cache
        cache = caches[0]
        with devices.autocast():
            cache[1] = function(
                shared.sd_model, required_prompts, steps,
                hires_steps, shared.opts.use_old_scheduling
            )

        if len(cache) > 2:
            cache[2] = modules.sd_hijack.model_hijack.extra_generation_params.copy()

        cache[0] = cache_params
        return cache[1]

    def setup_conds(self):
        """Setup conditioning for current batch."""
        prompts = prompt_parser.SdConditioning(
            self.prompts, width=self.width, height=self.height
        )
        negative_prompts = prompt_parser.SdConditioning(
            self.negative_prompts, width=self.width, height=self.height,
            is_negative_prompt=True
        )

        # Calculate total steps
        sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
        total_steps = sampler_config.total_steps(self.steps) if sampler_config else self.steps
        self.step_multiplier = total_steps // self.steps
        self.firstpass_steps = total_steps

        # Dynamic clip skip
        if opts.dynamic_clip_skip_enabled:
            self._setup_dynamic_clip_skip_conds(prompts, negative_prompts, total_steps)
        else:
            self._setup_standard_conds(prompts, negative_prompts, total_steps)

    def _setup_dynamic_clip_skip_conds(
        self,
        prompts: prompt_parser.SdConditioning,
        negative_prompts: prompt_parser.SdConditioning,
        total_steps: int
    ):
        """Setup conditioning with dynamic clip skip."""
        minimal_clip_skip = self._get_minimal_clip_skip()
        schedule = None
        custom_schedule_used = False

        if opts.dynamic_clip_skip_use_schedule:
            schedule = self._parse_dynamic_schedule(total_steps, minimal_clip_skip)
            custom_schedule_used = schedule is not None

        if schedule is None:
            start_clip_skip = max(int(opts.dynamic_clip_skip_start), MIN_CLIP_SKIP, minimal_clip_skip)
            start_clip_skip = min(start_clip_skip, MAX_CLIP_SKIP)
            schedule = self._build_dynamic_clip_skip_schedule(total_steps, start_clip_skip)
        else:
            start_clip_skip = schedule[0] if schedule else max(
                int(opts.dynamic_clip_skip_start), MIN_CLIP_SKIP, minimal_clip_skip
            )

        # Generate conditionings for each unique clip skip
        unique_clip_skips = list(dict.fromkeys(schedule))
        cond_sets = {}

        for clip_skip in unique_clip_skips:
            with temporary_clip_skip(clip_skip):
                uc = self.get_conds_with_caching(
                    prompt_parser.get_learned_conditioning, negative_prompts,
                    total_steps, [self.cached_uc], self.extra_network_data
                )
                c = self.get_conds_with_caching(
                    prompt_parser.get_multicond_learned_conditioning, prompts,
                    total_steps, [self.cached_c], self.extra_network_data
                )
                cond_sets[clip_skip] = (c, uc)

        self.dynamic_clip_skip_schedule = schedule
        self.dynamic_clip_skip_sets = cond_sets

        # Select initial conditioning
        selected_clip_skip = schedule[0] if schedule else opts.CLIP_stop_at_last_layers
        if selected_clip_skip in cond_sets:
            self.c, self.uc = cond_sets[selected_clip_skip]
        else:
            self._setup_standard_conds(prompts, negative_prompts, total_steps)
            return

        self.clip_skip = start_clip_skip
        if custom_schedule_used:
            self.extra_generation_params["Dynamic clip skip schedule"] = (
                opts.dynamic_clip_skip_schedule_values or ",".join(map(str, schedule))
            )
        else:
            self.extra_generation_params["Dynamic clip skip"] = f"{start_clip_skip}->2"
            self.extra_generation_params["Dynamic clip skip start"] = start_clip_skip

    def _setup_standard_conds(
        self,
        prompts: prompt_parser.SdConditioning,
        negative_prompts: prompt_parser.SdConditioning,
        total_steps: int
    ):
        """Setup standard conditioning without dynamic clip skip."""
        self.uc = self.get_conds_with_caching(
            prompt_parser.get_learned_conditioning, negative_prompts,
            total_steps, [self.cached_uc], self.extra_network_data
        )
        self.c = self.get_conds_with_caching(
            prompt_parser.get_multicond_learned_conditioning, prompts,
            total_steps, [self.cached_c], self.extra_network_data
        )
        self.clip_skip = opts.CLIP_stop_at_last_layers

    def get_conds(self) -> Tuple[Any, Any]:
        """Get current conditionings."""
        return self.c, self.uc

    def parse_extra_network_prompts(self):
        """Parse extra network prompts and extract data."""
        self.prompts, self.extra_network_data = extra_networks.parse_prompts(self.prompts)

    def save_samples(self) -> bool:
        """Check if samples should be saved to disk."""
        return (
            opts.samples_save and
            not self.do_not_save_samples and
            (opts.save_incomplete_images or
             (not state.interrupted and not state.skipped))
        )


# =============================================================================
# Processed Results
# =============================================================================

class Processed:
    """Container for processed results."""

    def __init__(
        self,
        p: StableDiffusionProcessing,
        images_list: List[Image.Image],
        seed: Union[int, List[int]] = -1,
        info: str = "",
        subseed: Optional[Union[int, List[int]]] = None,
        all_prompts: Optional[List[str]] = None,
        all_negative_prompts: Optional[List[str]] = None,
        all_seeds: Optional[List[int]] = None,
        all_subseeds: Optional[List[int]] = None,
        index_of_first_image: int = 0,
        infotexts: Optional[List[str]] = None,
        comments: str = "",
        extra_images_list: Optional[List[Image.Image]] = None
    ):
        """Initialize processed results."""
        self.images = images_list
        self.extra_images = extra_images_list or []
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = "".join(f"{comment}\n" for comment in p.comments)
        self.width = p.width
        self.height = p.height
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        self.sd_model_name = p.sd_model_name
        self.sd_model_hash = p.sd_model_hash
        self.sd_vae_name = p.sd_vae_name
        self.sd_vae_hash = p.sd_vae_hash
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = state.job_timestamp
        self.clip_skip = opts.CLIP_stop_at_last_layers
        self.token_merging_ratio = p.token_merging_ratio
        self.token_merging_ratio_hr = p.token_merging_ratio_hr
        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.s_min_uncond = p.s_min_uncond
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning

        # Normalize to single values
        self.prompt = self.prompt[0] if isinstance(self.prompt, list) else self.prompt
        self.negative_prompt = (
            self.negative_prompt[0]
            if isinstance(self.negative_prompt, list)
            else self.negative_prompt
        )
        self.seed = (
            int(self.seed[0]) if isinstance(self.seed, list) else int(self.seed)
        ) if self.seed is not None else -1
        self.subseed = (
            int(self.subseed[0]) if isinstance(self.subseed, list) else int(self.subseed)
        ) if self.subseed is not None else -1

        # Store lists
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = (
            all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        )
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info] * len(images_list)
        self.version = program_version()

    def js(self) -> str:
        """Return JSON representation."""
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_name": self.sd_model_name,
            "sd_model_hash": self.sd_model_hash,
            "sd_vae_name": self.sd_vae_name,
            "sd_vae_hash": self.sd_vae_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
            "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,
            "version": self.version,
        }
        return json.dumps(obj, default=lambda o: None)

    def infotext(self, p: StableDiffusionProcessing, index: int) -> str:
        """Generate infotext for specific image."""
        return create_infotext(
            p, self.all_prompts, self.all_seeds, self.all_subseeds,
            comments=[], position_in_batch=index % self.batch_size,
            iteration=index // self.batch_size
        )

    def get_token_merging_ratio(self, for_hr: bool = False) -> float:
        """Get token merging ratio."""
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio


# =============================================================================
# Helper Functions
# =============================================================================

def create_random_tensors(
    shape: Tuple[int, ...],
    seeds: List[int],
    subseeds: Optional[List[int]] = None,
    subseed_strength: float = 0.0,
    seed_resize_from_h: int = 0,
    seed_resize_from_w: int = 0,
    p: Optional[StableDiffusionProcessing] = None
) -> torch.Tensor:
    """Create random tensors for given seeds."""
    g = rng.ImageRNG(
        shape, seeds, subseeds=subseeds, subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w
    )
    return g.next()


class DecodedSamples(list):
    """Marker class for already decoded samples."""
    already_decoded = True


def decode_latent_batch(
    model: Any,
    batch: torch.Tensor,
    target_device: Optional[torch.device] = None,
    check_for_nans: bool = False
) -> DecodedSamples:
    """Decode batch of latents."""
    samples = DecodedSamples()
    samples_pytorch = decode_first_stage(model, batch).to(target_device)

    for x in samples_pytorch:
        samples.append(x)

    return samples


def create_infotext(
    p: StableDiffusionProcessing,
    all_prompts: List[str],
    all_seeds: List[int],
    all_subseeds: List[int],
    comments: Optional[List[str]] = None,
    iteration: int = 0,
    position_in_batch: int = 0,
    use_main_prompt: bool = False,
    index: Optional[int] = None,
    all_negative_prompts: Optional[List[str]] = None
) -> str:
    """Create infotext for generated image."""
    if use_main_prompt:
        index = 0
    elif index is None:
        index = position_in_batch + iteration * p.batch_size

    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    enable_hr = getattr(p, 'enable_hr', False)
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True)

    prompt_text = p.main_prompt if use_main_prompt else all_prompts[index]
    negative_prompt = (
        p.main_negative_prompt if use_main_prompt else all_negative_prompts[index]
    )

    uses_ensd = opts.eta_noise_seed_delta != 0
    if uses_ensd:
        uses_ensd = sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p)

    # Build generation parameters
    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "Schedule type": p.scheduler,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": p.all_seeds[0] if use_main_prompt else all_seeds[index],
        "Face restoration": opts.face_restoration_model if p.restore_faces else None,
        "Size": f"{p.width}x{p.height}",
        "Model hash": p.sd_model_hash if opts.add_model_hash_to_info else None,
        "Model": p.sd_model_name if opts.add_model_name_to_info else None,
        "FP8 weight": opts.fp8_storage if devices.fp8 else None,
        "Cache FP16 weight for LoRA": opts.cache_fp16_weight if devices.fp8 else None,
        "VAE hash": p.sd_vae_hash if opts.add_vae_hash_to_info else None,
        "VAE": p.sd_vae_name if opts.add_vae_name_to_info else None,
        "Variation seed": (
            None if p.subseed_strength == 0 else
            (p.all_subseeds[0] if use_main_prompt else all_subseeds[index])
        ),
        "Variation seed strength": (
            None if p.subseed_strength == 0 else p.subseed_strength
        ),
        "Seed resize from": (
            None if p.seed_resize_from_w <= 0 or p.seed_resize_from_h <= 0 else
            f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"
        ),
        "Denoising strength": p.extra_generation_params.get("Denoising strength"),
        "Conditional mask weight": (
            getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight)
            if p.is_using_inpainting_conditioning else None
        ),
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": opts.eta_noise_seed_delta if uses_ensd else None,
        "Token merging ratio": None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr": (
            None if not enable_hr or token_merging_ratio_hr == 0
            else token_merging_ratio_hr
        ),
        "Init image hash": getattr(p, 'init_img_hash', None),
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
        "Tiling": "True" if p.tiling else None,
        **p.extra_generation_params,
        "Version": program_version() if opts.add_version_to_infotext else None,
        "User": p.user if opts.add_user_name_to_info else None,
    }

    # Process generation parameters
    for key in list(generation_params.keys()):
        value = generation_params[key]
        try:
            if isinstance(value, list):
                generation_params[key] = value[index]
            elif callable(value):
                generation_params[key] = value(**locals())
        except Exception:
            errors.report(f'Error creating infotext for key "{key}"', exc_info=True)
            generation_params[key] = None

    # Format parameters
    params_text = ", ".join([
        k if k == v else f'{k}: {infotext_utils.quote(v)}'
        for k, v in generation_params.items()
        if v is not None
    ])

    negative_prompt_text = f"\nNegative prompt: {negative_prompt}" if negative_prompt else ""

    return f"{prompt_text}{negative_prompt_text}\n{params_text}".strip()


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_images(p: StableDiffusionProcessing) -> Processed:
    """Main entry point for processing images."""
    if p.scripts is not None:
        p.scripts.before_process(p)

    # Store original options
    stored_opts = {}
    for k in p.override_settings.keys():
        if k in opts.data:
            stored_opts[k] = opts.data[k]
        else:
            stored_opts[k] = opts.get_default(k)

    try:
        # Handle checkpoint override
        checkpoint_name = p.override_settings.get('sd_model_checkpoint')
        if checkpoint_name and sd_models.checkpoint_aliases.get(checkpoint_name) is None:
            p.override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()

        # Apply override settings
        for k, v in p.override_settings.items():
            opts.set(k, v, is_api=True, run_callbacks=False)

            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()
            elif k == 'sd_vae':
                sd_vae.reload_vae_weights()

        # Fix sampler and scheduler
        sd_samplers.fix_p_invalid_sampler_and_scheduler(p)

        # Process images
        with profiling.Profiler():
            res = process_images_inner(p)

    finally:
        # Restore options
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)
                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    return res


def _validate_processing_inputs(p: StableDiffusionProcessing):
    """Validate processing inputs."""
    if isinstance(p.prompt, list):
        if len(p.prompt) == 0:
            raise ValueError("Prompt list cannot be empty")
    elif p.prompt is None:
        raise ValueError("Prompt cannot be None")


def _setup_processing_state(p: StableDiffusionProcessing, seed: int, subseed: int):
    """Setup processing state and model info."""
    model = p.get_sd_model()

    # Set defaults from options
    if p.restore_faces is None:
        p.restore_faces = opts.face_restoration

    if p.tiling is None:
        p.tiling = opts.tiling

    # Handle refiner checkpoint
    if p.refiner_checkpoint not in (None, "", "None", "none"):
        p.refiner_checkpoint_info = sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
        if p.refiner_checkpoint_info is None:
            raise ValueError(f'Could not find checkpoint with name {p.refiner_checkpoint}')

    # Fix dimensions if needed
    if hasattr(model, 'fix_dimensions'):
        p.width, p.height = model.fix_dimensions(p.width, p.height)

    # Store model info
    p.sd_model_name = model.sd_checkpoint_info.name_for_extra
    p.sd_model_hash = model.sd_model_hash
    p.sd_vae_name = sd_vae.get_loaded_vae_name()
    p.sd_vae_hash = sd_vae.get_loaded_vae_hash()

    # Apply circular tiling
    apply_circular_forge(model, p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    # Setup
    p.fill_fields_from_opts()
    p.setup_prompts()

    # Setup seeds
    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [
            int(seed) + (x if p.subseed_strength == 0 else 0)
            for x in range(len(p.all_prompts))
        ]

    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    # Load embeddings
    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()


def _setup_batch(
    p: StableDiffusionProcessing,
    n: int
) -> Tuple[int, int]:
    """Setup batch for iteration n."""
    model = p.get_sd_model()
    model.forge_objects = model.forge_objects_original.shallow_copy()
    start_idx = n * p.batch_size
    end_idx = (n + 1) * p.batch_size

    p.prompts = p.all_prompts[start_idx:end_idx]
    p.negative_prompts = p.all_negative_prompts[start_idx:end_idx]
    p.seeds = p.all_seeds[start_idx:end_idx]
    p.subseeds = p.all_subseeds[start_idx:end_idx]

    # Create RNG
    latent_channels = getattr(model, 'latent_channels', LATENT_CHANNELS)
    p.rng = rng.ImageRNG(
        (latent_channels, p.height // VAE_SCALE_FACTOR, p.width // VAE_SCALE_FACTOR),
        p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
        seed_resize_from_h=p.seed_resize_from_h,
        seed_resize_from_w=p.seed_resize_from_w
    )

    return start_idx, end_idx


def _apply_ztsnr_if_needed(p: StableDiffusionProcessing):
    """Apply Zero Terminal SNR if needed."""
    model = p.get_sd_model()

    advanced_model_sampling_script = None
    if p.scripts is not None:
        for script in p.scripts.alwayson_scripts:
            if script.name == 'advanced model sampling for reforge':
                advanced_model_sampling_script = script
                break

    force_apply_ztsnr = False
    if advanced_model_sampling_script is not None:
        script_args = p.script_args[
            advanced_model_sampling_script.args_from:advanced_model_sampling_script.args_to
        ]
        if len(script_args) >= 4:
            force_apply_ztsnr = script_args[0] and script_args[3]

    should_apply = (
        opts.sd_noise_schedule == "Zero Terminal SNR" or
        getattr(model, 'ztsnr', False) or
        force_apply_ztsnr
    )

    if should_apply:
        model.sigmas_original = model.forge_objects.unet.model.model_sampling.sigmas
        model.alphas_cumprod_original = model.alphas_cumprod

        if not getattr(opts, 'use_old_clip_g_load_and_ztsnr_application', False):
            sd_models.apply_alpha_schedule_override(model, p, force_apply=force_apply_ztsnr)

        model.forge_objects.unet.model.model_sampling.set_sigmas(
            rescale_zero_terminal_snr_sigmas(
                model.forge_objects.unet.model.model_sampling.sigmas
            ).to(model.forge_objects.unet.model.device)
        )


def _restore_original_sigmas(p: StableDiffusionProcessing):
    """Restore original sigmas after ZTSNR."""
    model = p.get_sd_model()
    if hasattr(model, 'sigmas_original'):
        model.forge_objects.unet.model.model_sampling.set_sigmas(
            model.sigmas_original
        )
    if hasattr(model, 'alphas_cumprod_original'):
        model.alphas_cumprod = model.alphas_cumprod_original


def _process_decoded_samples(
    p: StableDiffusionProcessing,
    x_samples_ddim: torch.Tensor,
    n: int,
    start_idx: int,
    end_idx: int,
    output_images: List[Image.Image],
    infotexts: List[str]
):
    """Process decoded samples into final images."""
    save_samples = p.save_samples()

    def create_infotext_for_batch(index: int = 0, use_main_prompt: bool = False) -> str:
        """Create infotext for batch image."""
        if shared.opts.enable_prompt_comments_def:
            commented_prompts = p.prompts.copy()
            commented_prompts[index] = p.prompt
            commented_negative_prompts = p.negative_prompts.copy()
            commented_negative_prompts[index] = p.negative_prompt
            return create_infotext(
                p, commented_prompts, p.seeds, p.subseeds,
                use_main_prompt=False, index=index,
                all_negative_prompts=commented_negative_prompts
            )
        else:
            clean_prompts = [
                comments_parser.strip_comments(prompt)
                for prompt in p.prompts
            ]
            clean_negative_prompts = [
                comments_parser.strip_comments(negative_prompt)
                for negative_prompt in p.negative_prompts
            ]
            return create_infotext(
                p, clean_prompts, p.seeds, p.subseeds,
                use_main_prompt=False, index=index,
                all_negative_prompts=clean_negative_prompts
            )

    for i, x_sample in enumerate(x_samples_ddim):
        p.batch_index = i

        # Convert to numpy image
        x_sample_np = 255.0 * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample_np = x_sample_np.astype(np.uint8)

        # Face restoration
        if p.restore_faces:
            if save_samples and opts.save_images_before_face_restoration:
                before_image = Image.fromarray(x_sample_np)
                images.save_image(
                    before_image, p.outpath_samples, "",
                    p.seeds[i], p.prompts[i], opts.samples_format,
                    info=create_infotext_for_batch(i), p=p,
                    suffix="-before-face-restoration"
                )

            devices.torch_gc()
            x_sample_np = modules.face_restoration.restore_faces(x_sample_np)
            devices.torch_gc()

        image = Image.fromarray(x_sample_np)

        # Script postprocess image
        if p.scripts is not None:
            pp = scripts.PostprocessImageArgs(image)
            p.scripts.postprocess_image(p, pp)
            image = pp.image

        # Get mask and overlay
        mask_for_overlay = getattr(p, "mask_for_overlay", None)

        if not shared.opts.overlay_inpaint:
            overlay_image = None
        elif getattr(p, "overlay_images", None) is not None and i < len(p.overlay_images):
            overlay_image = p.overlay_images[i]
        else:
            overlay_image = None

        # Script postprocess mask/overlay
        if p.scripts is not None:
            ppmo = scripts.PostProcessMaskOverlayArgs(
                i, mask_for_overlay, overlay_image
            )
            p.scripts.postprocess_maskoverlay(p, ppmo)
            mask_for_overlay, overlay_image = ppmo.mask_for_overlay, ppmo.overlay_image

        # Apply color correction
        if p.color_corrections is not None and i < len(p.color_corrections):
            if save_samples and opts.save_images_before_color_correction:
                image_without_cc, _ = apply_overlay(image, p.paste_to, overlay_image)
                images.save_image(
                    image_without_cc, p.outpath_samples, "",
                    p.seeds[i], p.prompts[i], opts.samples_format,
                    info=create_infotext_for_batch(i), p=p,
                    suffix="-before-color-correction"
                )
            image = apply_color_correction(p.color_corrections[i], image)

        # Apply overlay
        image, original_denoised_image = apply_overlay(image, p.paste_to, overlay_image)

        p.pixels_after_sampling.append(image)

        # Script postprocess after composite
        if p.scripts is not None:
            pp = scripts.PostprocessImageArgs(image)
            p.scripts.postprocess_image_after_composite(p, pp)
            image = pp.image

        # Save image
        if save_samples:
            images.save_image(
                image, p.outpath_samples, "",
                p.seeds[i], p.prompts[i], opts.samples_format,
                info=create_infotext_for_batch(i), p=p
            )

        # Add metadata
        text = create_infotext_for_batch(i)
        infotexts.append(text)
        if opts.enable_pnginfo:
            image.info["parameters"] = text
        output_images.append(image)

        # Handle masks
        _handle_mask_outputs(
            p, i, mask_for_overlay, original_denoised_image, image,
            output_images, infotexts, create_infotext_for_batch, save_samples
        )


def _handle_mask_outputs(
    p: StableDiffusionProcessing,
    i: int,
    mask_for_overlay: Optional[Image.Image],
    original_denoised_image: Image.Image,
    image: Image.Image,
    output_images: List[Image.Image],
    infotexts: List[str],
    infotext_func: Callable,
    save_samples: bool
):
    """Handle mask-related output images."""
    if mask_for_overlay is None:
        return

    if opts.return_mask or opts.save_mask:
        image_mask = mask_for_overlay.convert('RGB')
        if save_samples and opts.save_mask:
            images.save_image(
                image_mask, p.outpath_samples, "",
                p.seeds[i], p.prompts[i], opts.samples_format,
                info=infotext_func(i), p=p, suffix="-mask"
            )
        if opts.return_mask:
            output_images.append(image_mask)

    if opts.return_mask_composite or opts.save_mask_composite:
        resized_mask = images.resize_image(
            2, mask_for_overlay, image.width, image.height
        ).convert('L')
        image_mask_composite = Image.composite(
            original_denoised_image.convert('RGBA').convert('RGBa'),
            Image.new('RGBa', image.size),
            resized_mask
        ).convert('RGBA')

        if save_samples and opts.save_mask_composite:
            images.save_image(
                image_mask_composite, p.outpath_samples, "",
                p.seeds[i], p.prompts[i], opts.samples_format,
                info=infotext_func(i), p=p, suffix="-mask-composite"
            )
        if opts.return_mask_composite:
            output_images.append(image_mask_composite)


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """Inner processing loop for both txt2img and img2img."""
    _validate_processing_inputs(p)

    devices.torch_gc()

    # Fix seeds
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    _setup_processing_state(p, seed, subseed)

    # Script processing
    if p.scripts is not None:
        p.scripts.process(p)

    # Process batches
    infotexts = []
    output_images = []

    model = p.get_sd_model()

    with torch.inference_mode():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # Load VAE approx for previews
            if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
                sd_vae_approx.model()

            sd_unet.apply_unet()

        # Setup job count
        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted or state.stopping_generation:
                break

            # Reload model weights (for refiner)
            sd_models.reload_model_weights()

            # Setup batch
            start_idx, end_idx = _setup_batch(p, n)

            # Script before batch
            if p.scripts is not None:
                p.scripts.before_process_batch(
                    p, batch_number=n, prompts=p.prompts,
                    seeds=p.seeds, subseeds=p.subseeds
                )

            if len(p.prompts) == 0:
                break

            # Parse extra networks
            p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                extra_networks.activate(p, p.extra_network_data)

            model.forge_objects = model.forge_objects_after_applying_lora.shallow_copy()

            # Script process batch
            if p.scripts is not None:
                p.scripts.process_batch(
                    p, batch_number=n, prompts=p.prompts,
                    seeds=p.seeds, subseeds=p.subseeds
                )

            # Setup conditioning
            p.setup_conds()
            p.extra_generation_params.update(model_hijack.extra_generation_params)

            # Save params.txt
            if n == 0 and not cmd_opts.no_prompt_history:
                params_path = os.path.join(paths.data_path, "params.txt")
                with open(params_path, "w", encoding="utf8") as file:
                    processed = Processed(p, [])
                    file.write(processed.infotext(p, 0))

            # Add comments
            for comment in model_hijack.comments:
                p.comment(comment)

            # Update state
            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            # Apply Zero Terminal SNR if needed
            _apply_ztsnr_if_needed(p)

            # Sample
            samples_ddim = p.sample(
                conditioning=p.c, unconditional_conditioning=p.uc,
                seeds=p.seeds, subseeds=p.subseeds,
                subseed_strength=p.subseed_strength, prompts=p.prompts
            )

            # Handle None samples (error case)
            if samples_ddim is None:
                logger.error("Sampling returned None. Skipping batch.")
                _restore_original_sigmas(p)
                state.nextjob()
                continue

            # Store latents
            for x_sample in samples_ddim:
                p.latents_after_sampling.append(x_sample)

            # Restore original sigmas
            _restore_original_sigmas(p)

            # Script post-sample
            if p.scripts is not None:
                ps = scripts.PostSampleArgs(samples_ddim)
                p.scripts.post_sample(p, ps)
                samples_ddim = ps.samples

            # Decode if needed
            if getattr(samples_ddim, 'already_decoded', False):
                x_samples_ddim = samples_ddim
            else:
                devices.test_for_nans(samples_ddim, "unet")

                if opts.sd_vae_decode_method != 'Full':
                    p.extra_generation_params['VAE Decoder'] = opts.sd_vae_decode_method

                x_samples_ddim = decode_latent_batch(
                    model, samples_ddim, target_device=devices.cpu, check_for_nans=True
                )

            # Normalize and clamp
            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim
            devices.torch_gc()

            state.nextjob()

            # Script postprocess batch
            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                # Re-fetch prompts for the batch
                p.prompts = p.all_prompts[start_idx:end_idx]
                p.negative_prompts = p.all_negative_prompts[start_idx:end_idx]

                batch_params = scripts.PostprocessBatchListArgs(list(x_samples_ddim))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                x_samples_ddim = batch_params.images

            # Process each image in batch
            _process_decoded_samples(
                p, x_samples_ddim, n, start_idx, end_idx, output_images, infotexts
            )

            del x_samples_ddim
            devices.torch_gc()

        # Ensure at least one infotext
        if not infotexts:
            infotexts.append(Processed(p, []).infotext(p, 0))

        p.color_corrections = None

        # Generate grid
        index_of_first_image = 0
        unwanted_grid = len(output_images) < 2 and opts.grid_only_if_multiple

        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = create_infotext(
                    p, p.all_prompts, p.all_seeds, p.all_subseeds,
                    use_main_prompt=True
                )
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(
                    grid, p.outpath_grids, "grid",
                    p.all_seeds[0], p.all_prompts[0], opts.grid_format,
                    info=create_infotext(
                        p, p.all_prompts, p.all_seeds, p.all_subseeds,
                        use_main_prompt=True
                    ),
                    short_filename=not opts.grid_extended_filename,
                    p=p, grid=True
                )

    # Deactivate extra networks
    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    devices.torch_gc()

    # Create result
    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
        extra_images_list=p.extra_result_images,
    )

    # Script postprocess
    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res


# =============================================================================
# Text-to-Image Processing
# =============================================================================

@dataclass(repr=False)
class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    """Processing class for text-to-image generation."""

    # Hi-res fix parameters
    enable_hr: bool = False
    denoising_strength: float = 0.75
    firstphase_width: int = 0
    firstphase_height: int = 0
    hr_scale: float = 2.0
    hr_upscaler: Optional[str] = None
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_checkpoint_name: Optional[str] = None
    hr_sampler_name: Optional[str] = None
    hr_scheduler: Optional[str] = None
    hr_prompt: str = ''
    hr_negative_prompt: str = ''
    hr_cfg: float = 1.0
    force_task_id: Optional[str] = None

    # Cached conditionings for HR pass - class level for persistence
    cached_hr_uc: List = field(default_factory=lambda: [None, None, None], init=False)
    cached_hr_c: List = field(default_factory=lambda: [None, None, None], init=False)

    # HR processing state
    hr_checkpoint_info: Optional[Dict] = field(default=None, init=False)
    hr_upscale_to_x: int = field(default=0, init=False)
    hr_upscale_to_y: int = field(default=0, init=False)
    truncate_x: int = field(default=0, init=False)
    truncate_y: int = field(default=0, init=False)
    applied_old_hires_behavior_to: Optional[Tuple[int, int]] = field(default=None, init=False)
    latent_scale_mode: Optional[Dict] = field(default=None, init=False)
    hr_c: Optional[Any] = field(default=None, init=False)
    hr_uc: Optional[Any] = field(default=None, init=False)
    all_hr_prompts: Optional[List[str]] = field(default=None, init=False)
    all_hr_negative_prompts: Optional[List[str]] = field(default=None, init=False)
    hr_prompts: Optional[List[str]] = field(default=None, init=False)
    hr_negative_prompts: Optional[List[str]] = field(default=None, init=False)
    hr_extra_network_data: Optional[Dict] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize txt2img processing."""
        super().__post_init__()

        # Handle first phase dimensions
        if self.firstphase_width != 0 or self.firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = self.firstphase_width
            self.height = self.firstphase_height

    def calculate_target_resolution(self):
        """Calculate target resolution for hi-res fix."""
        # Old hires fix behavior
        if opts.use_old_hires_fix_width_height and self.applied_old_hires_behavior_to != (self.width, self.height):
            self.hr_resize_x = self.width
            self.hr_resize_y = self.height
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height

            self.width, self.height = old_hires_fix_first_pass_dimensions(
                self.width, self.height
            )
            self.applied_old_hires_behavior_to = (self.width, self.height)

        # Calculate target resolution
        if self.hr_resize_x == 0 and self.hr_resize_y == 0:
            self.extra_generation_params["Hires upscale"] = self.hr_scale
            self.hr_upscale_to_x = int(self.width * self.hr_scale)
            self.hr_upscale_to_y = int(self.height * self.hr_scale)
        else:
            self.extra_generation_params["Hires resize"] = f"{self.hr_resize_x}x{self.hr_resize_y}"

            if self.hr_resize_y == 0:
                self.hr_upscale_to_x = self.hr_resize_x
                self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
            elif self.hr_resize_x == 0:
                self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                self.hr_upscale_to_y = self.hr_resize_y
            else:
                target_w, target_h = self.hr_resize_x, self.hr_resize_y
                src_ratio = self.width / self.height
                dst_ratio = self.hr_resize_x / self.hr_resize_y

                if src_ratio < dst_ratio:
                    self.hr_upscale_to_x = self.hr_resize_x
                    self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
                else:
                    self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                    self.hr_upscale_to_y = self.hr_resize_y

                self.truncate_x = (self.hr_upscale_to_x - target_w) // VAE_SCALE_FACTOR
                self.truncate_y = (self.hr_upscale_to_y - target_h) // VAE_SCALE_FACTOR

    def init(self, all_prompts: List[str], all_seeds: List[int], all_subseeds: List[int]):
        """Initialize txt2img processing with hi-res fix."""
        if self.enable_hr:
            self.extra_generation_params["Denoising strength"] = self.denoising_strength

            # HR checkpoint
            if self.hr_checkpoint_name and self.hr_checkpoint_name != 'Use same checkpoint':
                self.hr_checkpoint_info = sd_models.get_closet_checkpoint_match(
                    self.hr_checkpoint_name
                )
                if self.hr_checkpoint_info is None:
                    raise ValueError(
                        f'Could not find checkpoint with name {self.hr_checkpoint_name}'
                    )
                self.extra_generation_params["Hires checkpoint"] = self.hr_checkpoint_info.short_title

            # HR sampler
            if self.hr_sampler_name is not None and self.hr_sampler_name != self.sampler_name:
                self.extra_generation_params["Hires sampler"] = self.hr_sampler_name

            # HR prompt callbacks
            def get_hr_prompt(p, index, prompt_text, **kwargs):
                hr_prompt = p.all_hr_prompts[index]
                base_prompt = (
                    comments_parser.strip_comments(prompt_text)
                    if shared.opts.enable_prompt_comments_def else prompt_text
                )
                return hr_prompt if hr_prompt != base_prompt else None

            def get_hr_negative_prompt(p, index, negative_prompt, **kwargs):
                hr_negative_prompt = p.all_hr_negative_prompts[index]
                base_negative = (
                    comments_parser.strip_comments(negative_prompt)
                    if shared.opts.enable_prompt_comments_def else negative_prompt
                )
                return hr_negative_prompt if hr_negative_prompt != base_negative else None

            self.extra_generation_params["Hires prompt"] = get_hr_prompt
            self.extra_generation_params["Hires negative prompt"] = get_hr_negative_prompt
            self.extra_generation_params["Hires CFG Scale"] = self.hr_cfg
            self.extra_generation_params["Hires schedule type"] = None

            # HR scheduler
            if self.hr_scheduler is None:
                self.hr_scheduler = self.scheduler

            # Latent scale mode
            if self.hr_upscaler is not None:
                self.latent_scale_mode = shared.latent_upscale_modes.get(
                    self.hr_upscaler,
                    shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "nearest")
                )
                if self.latent_scale_mode is None and not any(
                    x.name == self.hr_upscaler for x in shared.sd_upscalers
                ):
                    raise ValueError(f"Could not find upscaler named {self.hr_upscaler}")

            self.calculate_target_resolution()

            # Update job count
            if not state.processing_has_refined_job_count:
                if state.job_count == -1:
                    state.job_count = self.n_iter

                if getattr(self, 'txt2img_upscale', False):
                    total_steps = (self.hr_second_pass_steps or self.steps) * state.job_count
                else:
                    total_steps = (
                        self.steps + (self.hr_second_pass_steps or self.steps)
                    ) * state.job_count

                shared.total_tqdm.updateTotal(total_steps)
                state.job_count = state.job_count * 2
                state.processing_has_refined_job_count = True

            # Additional parameters
            if self.hr_second_pass_steps:
                self.extra_generation_params["Hires steps"] = self.hr_second_pass_steps

            if self.hr_upscaler is not None:
                self.extra_generation_params["Hires upscaler"] = self.hr_upscaler

    def sample(
        self,
        conditioning: Any,
        unconditional_conditioning: Any,
        seeds: List[int],
        subseeds: List[int],
        subseed_strength: float,
        prompts: List[str]
    ) -> Optional[torch.Tensor]:
        """Perform txt2img sampling with optional hi-res fix."""
        model = self.get_sd_model()
        self.sampler = sd_samplers.create_sampler(self.sampler_name, model)

        # Use firstpass image if provided
        if self.firstpass_image is not None and self.enable_hr:
            return self._sample_with_firstpass_image(seeds, subseeds, subseed_strength, prompts)

        # Generate initial latents
        x = self.rng.next()

        model.forge_objects = model.forge_objects_after_applying_lora.shallow_copy()
        apply_token_merging(model, self.get_token_merging_ratio())

        # Script before sampling
        if self.scripts is not None:
            self.scripts.process_before_every_sampling(
                self, x=x, noise=x, c=conditioning, uc=unconditional_conditioning
            )

        # Apply modified noise if any
        if self.modified_noise is not None:
            x = self.modified_noise
            self.modified_noise = None

        # Sample
        samples = self.sampler.sample(
            self, x, conditioning, unconditional_conditioning,
            image_conditioning=self.txt2img_image_conditioning(x)
        )
        del x

        if not self.enable_hr:
            return samples

        devices.torch_gc()

        # Decode if needed for non-latent upscaling
        if self.latent_scale_mode is None:
            decoded_samples = torch.stack(
                decode_latent_batch(
                    model, samples, target_device=devices.cpu, check_for_nans=True
                )
            ).to(dtype=torch.float32)
        else:
            decoded_samples = None

        # Load HR checkpoint
        with sd_models.SkipWritingToConfig():
            sd_models.reload_model_weights(info=self.hr_checkpoint_info)

        return self.sample_hr_pass(
            samples, decoded_samples, seeds, subseeds, subseed_strength, prompts
        )

    def _sample_with_firstpass_image(
        self,
        seeds: List[int],
        subseeds: List[int],
        subseed_strength: float,
        prompts: List[str]
    ) -> Optional[torch.Tensor]:
        """Sample using a provided firstpass image."""
        model = self.get_sd_model()

        if self.latent_scale_mode is None:
            # Use decoded image
            image = np.array(self.firstpass_image).astype(np.float32) / 255.0 * 2.0 - 1.0
            image = np.moveaxis(image, 2, 0)
            samples = None
            decoded_samples = torch.asarray(np.expand_dims(image, 0))
        else:
            # Encode to latents
            image = np.array(self.firstpass_image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(np.expand_dims(image, axis=0))
            image = image.to(shared.device, dtype=torch.float32)

            if opts.sd_vae_encode_method != 'Full':
                self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

            samples = images_tensor_to_samples(
                image, approximation_indexes.get(opts.sd_vae_encode_method), model
            )
            decoded_samples = None
            devices.torch_gc()

        # Load HR checkpoint
        with sd_models.SkipWritingToConfig():
            sd_models.reload_model_weights(info=self.hr_checkpoint_info)

        return self.sample_hr_pass(
            samples, decoded_samples, seeds, subseeds, subseed_strength, prompts
        )

    def sample_hr_pass(
        self,
        samples: Optional[torch.Tensor],
        decoded_samples: Optional[torch.Tensor],
        seeds: List[int],
        subseeds: List[int],
        subseed_strength: float,
        prompts: List[str]
    ) -> Optional[torch.Tensor]:
        """Perform hi-res fix pass."""
        if shared.state.interrupted:
            return samples if samples is not None else decoded_samples

        model = self.get_sd_model()
        self.is_hr_pass = True
        target_width = self.hr_upscale_to_x
        target_height = self.hr_upscale_to_y

        # Create HR sampler
        img2img_sampler_name = self.hr_sampler_name or self.sampler_name
        self.sampler = sd_samplers.create_sampler(img2img_sampler_name, model)

        # Upscale
        if self.latent_scale_mode is not None and samples is not None:
            samples, image_conditioning = self._upscale_latent(
                samples, target_width, target_height, seeds, prompts
            )
        elif decoded_samples is not None:
            samples, image_conditioning = self._upscale_image(
                decoded_samples, target_width, target_height, seeds, prompts
            )
        else:
            logger.error("No samples or decoded samples available for HR pass.")
            self.is_hr_pass = False
            return None

        shared.state.nextjob()

        # Truncate if needed
        if self.truncate_x > 0 or self.truncate_y > 0:
            samples = samples[
                :, :,
                self.truncate_y // 2:samples.shape[2] - (self.truncate_y + 1) // 2,
                self.truncate_x // 2:samples.shape[3] - (self.truncate_x + 1) // 2
            ]

        # Prepare noise for img2img
        self.rng = rng.ImageRNG(
            samples.shape[1:], self.seeds, subseeds=self.subseeds,
            subseed_strength=self.subseed_strength,
            seed_resize_from_h=self.seed_resize_from_h,
            seed_resize_from_w=self.seed_resize_from_w
        )
        noise = self.rng.next()

        devices.torch_gc()

        # Activate extra networks for HR
        if not self.disable_extra_networks:
            with devices.autocast():
                extra_networks.activate(self, self.hr_extra_network_data)

        # Calculate HR conditioning
        with devices.autocast():
            self.calculate_hr_conds()

        # Apply token merging
        model.forge_objects = model.forge_objects_after_applying_lora.shallow_copy()
        apply_token_merging(model, self.get_token_merging_ratio(for_hr=True))

        # Script before HR sampling
        if self.scripts is not None:
            self.scripts.before_hr(self)
            self.scripts.process_before_every_sampling(
                p=self, x=samples, noise=noise, c=self.hr_c, uc=self.hr_uc
            )

        # Apply modified noise
        if self.modified_noise is not None:
            noise = self.modified_noise
            self.modified_noise = None

        # Sample HR pass
        samples = self.sampler.sample_img2img(
            self, samples, noise, self.hr_c, self.hr_uc,
            steps=self.hr_second_pass_steps or self.steps,
            image_conditioning=image_conditioning
        )

        self.sampler = None
        devices.torch_gc()

        # Decode results
        decoded_samples = decode_latent_batch(
            model, samples, target_device=devices.cpu, check_for_nans=True
        )

        self.is_hr_pass = False
        return decoded_samples

    def _upscale_latent(
        self,
        samples: torch.Tensor,
        target_width: int,
        target_height: int,
        seeds: List[int],
        prompts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upscale using latent space interpolation."""
        model = self.get_sd_model()

        for i in range(samples.shape[0]):
            self._save_intermediate(samples, i, seeds, prompts)

        samples = torch.nn.functional.interpolate(
            samples,
            size=(target_height // VAE_SCALE_FACTOR, target_width // VAE_SCALE_FACTOR),
            mode=self.latent_scale_mode["mode"],
            antialias=self.latent_scale_mode["antialias"]
        )

        # Generate conditioning
        mask_weight = getattr(
            self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight
        )
        if mask_weight < 1.0:
            image_conditioning = self.img2img_image_conditioning(
                decode_first_stage(model, samples), samples
            )
        else:
            image_conditioning = self.txt2img_image_conditioning(samples)

        return samples, image_conditioning

    def _upscale_image(
        self,
        decoded_samples: torch.Tensor,
        target_width: int,
        target_height: int,
        seeds: List[int],
        prompts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upscale using image-space upscaler."""
        lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
        batch_images = []

        for i, x_sample in enumerate(lowres_samples):
            x_sample_np = 255.0 * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
            x_sample_np = x_sample_np.astype(np.uint8)
            image = Image.fromarray(x_sample_np)

            self._save_intermediate(image, i, seeds, prompts)

            image = images.resize_image(
                0, image, target_width, target_height, upscaler_name=self.hr_upscaler
            )
            image_np = np.array(image).astype(np.float32) / 255.0
            image_np = np.moveaxis(image_np, 2, 0)
            batch_images.append(image_np)

        decoded_samples = torch.from_numpy(np.array(batch_images))
        decoded_samples = decoded_samples.to(shared.device, dtype=torch.float32)

        if opts.sd_vae_encode_method != 'Full':
            self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

        samples = images_tensor_to_samples(
            decoded_samples, approximation_indexes.get(opts.sd_vae_encode_method)
        )
        image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

        return samples, image_conditioning

    def _save_intermediate(
        self,
        image_or_samples: Union[Image.Image, torch.Tensor],
        index: int,
        seeds: List[int],
        prompts: List[str]
    ):
        """Save intermediate image before hi-res fix."""
        if not self.save_samples() or not opts.save_images_before_highres_fix:
            return

        if isinstance(image_or_samples, torch.Tensor):
            image = sd_samplers.sample_to_image(image_or_samples, index, approximation=0)
        else:
            image = image_or_samples

        info = create_infotext(
            self, self.all_prompts, self.all_seeds, self.all_subseeds,
            [], iteration=self.iteration, position_in_batch=index
        )
        images.save_image(
            image, self.outpath_samples, "", seeds[index], prompts[index],
            opts.samples_format, info=info, p=self, suffix="-before-highres-fix"
        )

    def close(self):
        """Clean up HR resources."""
        super().close()
        self.hr_c = None
        self.hr_uc = None
        if not opts.persistent_cond_cache:
            StableDiffusionProcessingTxt2Img.cached_hr_uc = [None, None, None]
            StableDiffusionProcessingTxt2Img.cached_hr_c = [None, None, None]

    def setup_prompts(self):
        """Setup prompts including HR prompts."""
        super().setup_prompts()

        if not self.enable_hr:
            return

        # Default HR prompts
        if self.hr_prompt == '':
            self.hr_prompt = self.prompt
        if self.hr_negative_prompt == '':
            self.hr_negative_prompt = self.negative_prompt

        # Create HR prompt lists
        if isinstance(self.hr_prompt, list):
            self.all_hr_prompts = self.hr_prompt
        else:
            self.all_hr_prompts = self.batch_size * self.n_iter * [self.hr_prompt]

        if isinstance(self.hr_negative_prompt, list):
            self.all_hr_negative_prompts = self.hr_negative_prompt
        else:
            self.all_hr_negative_prompts = self.batch_size * self.n_iter * [self.hr_negative_prompt]

        # Apply styles
        self.all_hr_prompts = [
            shared.prompt_styles.apply_styles_to_prompt(x, self.styles)
            for x in self.all_hr_prompts
        ]
        self.all_hr_negative_prompts = [
            shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles)
            for x in self.all_hr_negative_prompts
        ]

    def calculate_hr_conds(self):
        """Calculate conditioning for HR pass."""
        if self.hr_c is not None:
            return

        hr_prompts = prompt_parser.SdConditioning(
            self.hr_prompts, width=self.hr_upscale_to_x, height=self.hr_upscale_to_y
        )
        hr_negative_prompts = prompt_parser.SdConditioning(
            self.hr_negative_prompts, width=self.hr_upscale_to_x,
            height=self.hr_upscale_to_y, is_negative_prompt=True
        )

        # Calculate total steps
        sampler_config = sd_samplers.find_sampler_config(
            self.hr_sampler_name or self.sampler_name
        )
        steps = self.hr_second_pass_steps or self.steps
        total_steps = sampler_config.total_steps(steps) if sampler_config else steps

        # HR CFG handling
        if self.hr_cfg < 0 or (self.hr_cfg == 0 and self.cfg_scale == 0):
            self.hr_uc = None
            logger.info(
                'Skipping unconditional conditioning (HR pass) '
                'due to negative HR CFG or zero CFG scales.'
            )
            actual_hr_cfg = 0
        elif self.hr_cfg == 0:
            self.hr_cfg = self.cfg_scale
            actual_hr_cfg = self.cfg_scale
            self.hr_uc = self.get_conds_with_caching(
                prompt_parser.get_learned_conditioning, hr_negative_prompts,
                self.firstpass_steps, [self.cached_hr_uc, self.cached_uc],
                self.hr_extra_network_data, total_steps
            )
        else:
            actual_hr_cfg = self.hr_cfg
            self.hr_uc = self.get_conds_with_caching(
                prompt_parser.get_learned_conditioning, hr_negative_prompts,
                self.firstpass_steps, [self.cached_hr_uc, self.cached_uc],
                self.hr_extra_network_data, total_steps
            )

        if self.extra_generation_params.get("Hires CFG Scale", None) == 0:
            self.extra_generation_params["Hires CFG Scale"] = actual_hr_cfg

        # Get HR conditioning
        self.hr_c = self.get_conds_with_caching(
            prompt_parser.get_multicond_learned_conditioning, hr_prompts,
            self.firstpass_steps, [self.cached_hr_c, self.cached_c],
            self.hr_extra_network_data, total_steps
        )

    def setup_conds(self):
        """Setup conditioning with HR support."""
        if self.is_hr_pass:
            # In HR pass, use HR conditioning
            self.hr_c = None
            self.calculate_hr_conds()
            return

        super().setup_conds()

        self.hr_uc = None
        self.hr_c = None

        # Pre-calculate HR conditioning if needed
        if self.enable_hr and self.hr_checkpoint_info is None:
            if shared.opts.hires_fix_use_firstpass_conds:
                self.calculate_hr_conds()
            elif shared.sd_model.sd_checkpoint_info == sd_models.select_checkpoint():
                # Calculate before unloading cond NN in lowvram mode
                with devices.autocast():
                    extra_networks.activate(self, self.hr_extra_network_data)

                self.calculate_hr_conds()

                with devices.autocast():
                    extra_networks.activate(self, self.extra_network_data)

    def get_conds(self) -> Tuple[Any, Any]:
        """Get current conditionings."""
        if self.is_hr_pass:
            return self.hr_c, self.hr_uc
        return super().get_conds()

    def parse_extra_network_prompts(self):
        """Parse extra network prompts including HR prompts."""
        res = super().parse_extra_network_prompts()

        if self.enable_hr:
            start_idx = self.iteration * self.batch_size
            end_idx = (self.iteration + 1) * self.batch_size

            self.hr_prompts = self.all_hr_prompts[start_idx:end_idx]
            self.hr_negative_prompts = self.all_hr_negative_prompts[start_idx:end_idx]

            self.hr_prompts, self.hr_extra_network_data = extra_networks.parse_prompts(
                self.hr_prompts
            )

        return res


# =============================================================================
# Image-to-Image Processing
# =============================================================================

@dataclass(repr=False)
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    """Processing class for image-to-image generation."""

    # Input parameters
    init_images: Optional[List[Image.Image]] = None
    resize_mode: int = 0
    denoising_strength: float = 0.75
    image_cfg_scale: Optional[float] = None
    mask: Optional[Any] = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    _mask_blur: Optional[int] = field(default=None, init=False)
    mask_round: bool = True
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    initial_noise_multiplier: Optional[float] = None
    latent_mask: Optional[Image.Image] = None
    force_task_id: Optional[str] = None

    # Processing state
    image_mask: Optional[Image.Image] = field(default=None, init=False)
    nmask: Optional[torch.Tensor] = field(default=None, init=False)
    image_conditioning: Optional[torch.Tensor] = field(default=None, init=False)
    init_img_hash: Optional[str] = field(default=None, init=False)
    mask_for_overlay: Optional[Image.Image] = field(default=None, init=False)
    init_latent: Optional[torch.Tensor] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize img2img processing."""
        super().__post_init__()

        self.image_mask = self.mask
        self.mask = None
        self.initial_noise_multiplier = (
            self.initial_noise_multiplier or opts.initial_noise_multiplier
        )

    @property
    def mask_blur(self) -> Optional[int]:
        """Get uniform mask blur value."""
        if self.mask_blur_x == self.mask_blur_y:
            return self.mask_blur_x
        return self._mask_blur

    @mask_blur.setter
    def mask_blur(self, value: Optional[int]):
        """Set uniform mask blur value."""
        if value is not None and isinstance(value, int):
            self.mask_blur_x = value
            self.mask_blur_y = value
        self._mask_blur = value

    def init(self, all_prompts: List[str], all_seeds: List[int], all_subseeds: List[int]):
        """Initialize img2img processing."""
        model = self.get_sd_model()
        self.extra_generation_params["Denoising strength"] = self.denoising_strength

        # Set image CFG scale for edit models
        self.image_cfg_scale = (
            self.image_cfg_scale
            if model.cond_stage_key == "edit" else None
        )

        # Create sampler
        self.sampler = sd_samplers.create_sampler(self.sampler_name, model)

        # Process mask and images
        crop_region = None
        image_mask = self.image_mask

        if image_mask is not None:
            image_mask, crop_region = self._process_mask(image_mask)

        # Determine latent mask
        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        # Script before init images
        if self.scripts is not None:
            self.scripts.before_process_init_images(
                self, dict(crop_region=crop_region, image_mask=image_mask)
            )

        # Process init images
        batch_images, image_mask = self._process_init_images(
            image_mask, crop_region, latent_mask
        )

        # Encode to latents
        image_tensor = torch.from_numpy(batch_images).to(shared.device, dtype=torch.float32)

        if opts.sd_vae_encode_method != 'Full':
            self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

        self.init_latent = images_tensor_to_samples(
            image_tensor, approximation_indexes.get(opts.sd_vae_encode_method), model
        )
        devices.torch_gc()

        # Resize if needed
        if self.resize_mode == 3:
            self.init_latent = torch.nn.functional.interpolate(
                self.init_latent,
                size=(self.height // VAE_SCALE_FACTOR, self.width // VAE_SCALE_FACTOR),
                mode="bilinear"
            )

        # Process mask for latents
        if image_mask is not None:
            self._setup_latent_mask(image_mask, latent_mask, all_seeds)

        # Generate image conditioning
        self.image_conditioning = self.img2img_image_conditioning(
            image_tensor * 2 - 1, self.init_latent, image_mask, self.mask_round
        )

    def _process_mask(
        self,
        image_mask: Image.Image
    ) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
        """Process the inpainting mask."""
        crop_region = None

        # Create binary mask
        image_mask = create_binary_mask(image_mask, round_mask=self.mask_round)

        # Invert mask if needed
        if self.inpainting_mask_invert:
            image_mask = ImageOps.invert(image_mask)
            self.extra_generation_params["Mask mode"] = "Inpaint not masked"

        # Apply blur
        if self.mask_blur_x > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
            image_mask = Image.fromarray(np_mask)

        if self.mask_blur_y > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
            image_mask = Image.fromarray(np_mask)

        if self.mask_blur_x > 0 or self.mask_blur_y > 0:
            self.extra_generation_params["Mask blur"] = self.mask_blur

        # Full resolution inpaint
        if self.inpaint_full_res:
            self.mask_for_overlay = image_mask
            mask = image_mask.convert('L')
            crop_region = masking.get_crop_region_v2(mask, self.inpaint_full_res_padding)

            if crop_region:
                crop_region = masking.expand_crop_region(
                    crop_region, self.width, self.height, mask.width, mask.height
                )
                x1, y1, x2, y2 = crop_region
                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2 - x1, y2 - y1)
                self.extra_generation_params["Inpaint area"] = "Only masked"
                self.extra_generation_params["Masked area padding"] = self.inpaint_full_res_padding
            else:
                crop_region = None
                image_mask = None
                self.mask_for_overlay = None
                self.inpaint_full_res = False
                message = (
                    'Unable to perform "Inpaint Only mask" because mask is blank, '
                    'switch to img2img mode.'
                )
                model_hijack.comments.append(message)
                logger.info(message)
        else:
            # Resize mask
            image_mask = images.resize_image(
                self.resize_mode, image_mask, self.width, self.height
            )
            np_mask = np.array(image_mask)
            np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
            self.mask_for_overlay = Image.fromarray(np_mask)

        self.overlay_images = []

        return image_mask, crop_region

    def _process_init_images(
        self,
        image_mask: Optional[Image.Image],
        crop_region: Optional[Tuple[int, int, int, int]],
        latent_mask: Optional[Image.Image]
    ) -> Tuple[np.ndarray, Optional[Image.Image]]:
        """Process init images into batch array."""
        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []

        imgs = []
        for img in self.init_images:
            # Save init image
            if opts.save_init_img:
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                images.save_image(
                    img, path=opts.outdir_init_images, basename=None,
                    forced_filename=self.init_img_hash, save_to_dirs=False,
                    existing_info=img.info
                )

            # Flatten and resize
            image = images.flatten(img, opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            # Apply mask
            if image_mask is not None:
                if self.mask_for_overlay.size != (image.width, image.height):
                    self.mask_for_overlay = images.resize_image(
                        self.resize_mode, self.mask_for_overlay, image.width, image.height
                    )

                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(
                    image.convert("RGBA").convert("RGBa"),
                    mask=ImageOps.invert(self.mask_for_overlay.convert('L'))
                )
                self.overlay_images.append(image_masked.convert('RGBA'))

            # Crop if needed
            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            # Fill masked area
            if image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)
                    if self.inpainting_fill == 0:
                        self.extra_generation_params["Masked content"] = 'fill'

            # Setup color correction
            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))

            # Convert to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_np = np.moveaxis(image_np, 2, 0)
            imgs.append(image_np)

        # Create batch tensor
        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size
            if self.color_corrections is not None and len(self.color_corrections) == 1:
                self.color_corrections = self.color_corrections * self.batch_size
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(
                f"Too many images passed: {len(imgs)}; expecting {self.batch_size} or less"
            )

        return batch_images, image_mask

    def _setup_latent_mask(
        self,
        image_mask: Image.Image,
        latent_mask: Image.Image,
        all_seeds: List[int]
    ):
        """Setup mask tensors for latent space."""
        init_mask = latent_mask
        latmask = init_mask.convert('RGB').resize(
            (self.init_latent.shape[3], self.init_latent.shape[2])
        )
        latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
        latmask = latmask[0]

        if self.mask_round:
            latmask = np.around(latmask)

        latmask = np.tile(latmask[None], (self.init_latent.shape[1], 1, 1))

        self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(devices.dtype)
        self.nmask = torch.asarray(latmask).to(shared.device).type(devices.dtype)

        # Apply mask fill
        if self.inpainting_fill == 2:
            self.init_latent = (
                self.init_latent * self.mask +
                create_random_tensors(
                    self.init_latent.shape[1:],
                    all_seeds[0:self.init_latent.shape[0]]
                ) * self.nmask
            )
            self.extra_generation_params["Masked content"] = 'latent noise'
        elif self.inpainting_fill == 3:
            self.init_latent = self.init_latent * self.mask
            self.extra_generation_params["Masked content"] = 'latent nothing'

    def sample(
        self,
        conditioning: Any,
        unconditional_conditioning: Any,
        seeds: List[int],
        subseeds: List[int],
        subseed_strength: float,
        prompts: List[str]
    ) -> Optional[torch.Tensor]:
        """Perform img2img sampling."""
        model = self.get_sd_model()
        x = self.rng.next()

        # Apply noise multiplier
        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            x *= self.initial_noise_multiplier

        # Apply token merging
        model.forge_objects = model.forge_objects_after_applying_lora.shallow_copy()
        apply_token_merging(model, self.get_token_merging_ratio())

        # Script before sampling
        if self.scripts is not None:
            self.scripts.process_before_every_sampling(
                self, x=self.init_latent, noise=x,
                c=conditioning, uc=unconditional_conditioning
            )

        # Apply modified noise
        if self.modified_noise is not None:
            x = self.modified_noise
            self.modified_noise = None

        # Sample
        samples = self.sampler.sample_img2img(
            self, self.init_latent, x, conditioning, unconditional_conditioning,
            image_conditioning=self.image_conditioning
        )

        # Apply mask blend
        if self.mask is not None:
            blended_samples = samples * self.nmask + self.init_latent * self.mask

            # Script mask blend
            if self.scripts is not None:
                mba = scripts.MaskBlendArgs(
                    samples, self.nmask, self.init_latent, self.mask, blended_samples
                )
                self.scripts.on_mask_blend(self, mba)
                blended_samples = mba.blended_latent

            samples = blended_samples

        del x
        devices.torch_gc()

        return samples

    def get_token_merging_ratio(self, for_hr: bool = False) -> float:
        """Get token merging ratio for img2img."""
        return (
            self.token_merging_ratio or
            ("token_merging_ratio" in self.override_settings and opts.token_merging_ratio) or
            opts.token_merging_ratio_img2img or
            opts.token_merging_ratio
        )
