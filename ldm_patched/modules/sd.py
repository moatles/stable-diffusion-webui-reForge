from __future__ import annotations
import json
import torch
from enum import Enum
import logging

from ldm_patched.modules import model_management
from ldm_patched.modules.utils import ProgressBar
from ldm_patched.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
from ldm_patched.ldm.cascade.stage_a import StageA
from ldm_patched.ldm.cascade.stage_c_coder import StageC_coder
from ldm_patched.ldm.audio.autoencoder import AudioOobleckVAE
import ldm_patched.ldm.genmo.vae.model
import ldm_patched.ldm.lightricks.vae.causal_video_autoencoder
import ldm_patched.ldm.cosmos.vae
import ldm_patched.ldm.wan.vae
import ldm_patched.ldm.hunyuan3d.vae
import ldm_patched.ldm.ace.vae.music_dcae_pipeline
import yaml
import math
import os

import ldm_patched.modules.utils

from . import clip_vision
from . import gligen
from . import diffusers_convert
from . import model_detection

from . import sd1_clip
from . import sdxl_clip
import ldm_patched.modules.text_encoders.sd2_clip
import ldm_patched.modules.text_encoders.sd3_clip
import ldm_patched.modules.text_encoders.sa_t5
import ldm_patched.modules.text_encoders.aura_t5
import ldm_patched.modules.text_encoders.pixart_t5
import ldm_patched.modules.text_encoders.hydit
import ldm_patched.modules.text_encoders.flux
import ldm_patched.modules.text_encoders.long_clipl
import ldm_patched.modules.text_encoders.genmo
import ldm_patched.modules.text_encoders.lt
import ldm_patched.modules.text_encoders.hunyuan_video
import ldm_patched.modules.text_encoders.cosmos
import ldm_patched.modules.text_encoders.lumina2
import ldm_patched.modules.text_encoders.wan
import ldm_patched.modules.text_encoders.hidream
import ldm_patched.modules.text_encoders.ace
import ldm_patched.modules.text_encoders.omnigen2

import ldm_patched.modules.text_encoders
import ldm_patched.modules.lora
import ldm_patched.modules.lora_convert
import ldm_patched.hooks
import ldm_patched.t2ia.adapter
import ldm_patched.taesd.taesd

import ldm_patched.ldm.flux.redux
from modules import shared

def load_model_weights(model, sd):
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        logging.warning("missing {}".format(m))
    return model

def load_clip_weights(model, sd):
    k = list(sd.keys())
    for x in k:
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            sd[y] = sd.pop(x)

    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
        ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
        if ids.dtype == torch.float32:
            sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

    sd = ldm_patched.modules.utils.transformers_convert(sd, "cond_stage_model.model.", "cond_stage_model.transformer.text_model.", 24)
    return load_model_weights(model, sd)

def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename='default'):
    model_flag = type(model.model).__name__ if model is not None else 'default'

    key_map = {}
    if model is not None and strength_model != 0:
        key_map = ldm_patched.modules.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None and strength_clip != 0:
        key_map = ldm_patched.modules.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    if not key_map:
        return (model, clip)

    lora = ldm_patched.modules.lora_convert.convert_lora(lora)
    loaded = ldm_patched.modules.lora.load_lora(lora, key_map)

    if model is not None:
        new_modelpatcher = model.clone()
        loaded_keys_unet = new_modelpatcher.add_patches(loaded, strength_model)
    else:
        new_modelpatcher = None
        loaded_keys_unet = set()

    if clip is not None:
        new_clip = clip.clone()
        loaded_keys_clip = new_clip.add_patches(loaded, strength_clip)
    else:
        new_clip = None
        loaded_keys_clip = set()

    loaded_keys_unet = set(loaded_keys_unet)
    loaded_keys_clip = set(loaded_keys_clip)

    for x in loaded:
        if (x not in loaded_keys_unet) and (x not in loaded_keys_clip):
            logging.warning("NOT LOADED {}".format(x))

    if loaded_keys_unet or loaded_keys_clip:
        total_loaded_keys = len(loaded_keys_unet) + len(loaded_keys_clip)
        print(f'[LORA] Loaded {filename} for {model_flag} with {total_loaded_keys} keys (UNet: {len(loaded_keys_unet)}, CLIP: {len(loaded_keys_clip)}) at weight {strength_clip}')

    return (new_modelpatcher, new_clip)


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False, tokenizer_data={}, parameters=0, model_options={}):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_options.get("load_device", model_management.text_encoder_device())
        offload_device = model_options.get("offload_device", model_management.text_encoder_offload_device())
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = model_management.text_encoder_dtype(load_device)

        params['dtype'] = dtype
        params['device'] = model_options.get("initial_device", model_management.text_encoder_initial_device(load_device, offload_device, parameters * model_management.dtype_size(dtype)))
        params['model_options'] = model_options

        self.cond_stage_model = clip(**(params))

        if shared.opts.cond_stage_model_device_compatibility_check:
            for dt in self.cond_stage_model.dtypes:
                if not model_management.supports_cast(load_device, dt):
                    load_device = offload_device
                    if params['device'] != offload_device:
                        self.cond_stage_model.to(offload_device)
                        logging.warning("Had to shift TE back.")
                    print(f"Conditional stage model dtype {dt} not supported. Falling back to {offload_device}.")
                    break
        else:
            logging.info("Conditional stage model device compatibility check is disabled.")

        self.tokenizer = tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.patcher = ldm_patched.modules.model_patcher.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.patcher.hook_mode = ldm_patched.hooks.EnumHookMode.MinVram
        self.patcher.is_clip = True
        self.apply_hooks_to_conds = None
        if params['device'] == load_device:
            model_management.load_models_gpu([self.patcher], force_full_load=True)
        self.layer_idx = None
        self.use_clip_schedule = False
        logging.info("CLIP/text encoder model load device: {}, offload device: {}, current: {}, dtype: {}".format(load_device, offload_device, params['device'], dtype))
        self.tokenizer_options = {}

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        n.tokenizer_options = self.tokenizer_options.copy()
        n.use_clip_schedule = self.use_clip_schedule
        n.apply_hooks_to_conds = self.apply_hooks_to_conds
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def set_tokenizer_option(self, option_name, value):
        self.tokenizer_options[option_name] = value

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False, **kwargs):
        tokenizer_options = kwargs.get("tokenizer_options", {})
        if len(self.tokenizer_options) > 0:
            tokenizer_options = {**self.tokenizer_options, **tokenizer_options}
        if len(tokenizer_options) > 0:
            kwargs["tokenizer_options"] = tokenizer_options
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def add_hooks_to_dict(self, pooled_dict: dict[str]):
        if self.apply_hooks_to_conds:
            pooled_dict["hooks"] = self.apply_hooks_to_conds
        return pooled_dict

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict: dict[str]={}, show_pbar=True):
        all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
        all_hooks = self.patcher.forced_hooks
        if all_hooks is None or not self.use_clip_schedule:
            return_pooled = "unprojected" if unprojected else True
            pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)
            cond = pooled_dict.pop("cond")
            pooled_dict.update(add_dict)
            all_cond_pooled.append([cond, pooled_dict])
        else:
            scheduled_keyframes = all_hooks.get_hooks_for_clip_schedule()

            self.cond_stage_model.reset_clip_options()
            if self.layer_idx is not None:
                self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
            if unprojected:
                self.cond_stage_model.set_clip_options({"projected_pooled": False})

            self.load_model()
            all_hooks.reset()
            self.patcher.patch_hooks(None)
            if show_pbar:
                pbar = ProgressBar(len(scheduled_keyframes))

            for scheduled_opts in scheduled_keyframes:
                t_range = scheduled_opts[0]
                if "start_percent" in add_dict:
                    if t_range[1] < add_dict["start_percent"]:
                        continue
                if "end_percent" in add_dict:
                    if t_range[0] > add_dict["end_percent"]:
                        continue
                hooks_keyframes = scheduled_opts[1]
                for hook, keyframe in hooks_keyframes:
                    hook.hook_keyframe._current_keyframe = keyframe
                self.patcher.patch_hooks(all_hooks)
                o = self.cond_stage_model.encode_token_weights(tokens)
                cond, pooled = o[:2]
                pooled_dict = {"pooled_output": pooled}
                pooled_dict["clip_start_percent"] = t_range[0]
                pooled_dict["clip_end_percent"] = t_range[1]
                pooled_dict.update(add_dict)
                self.add_hooks_to_dict(pooled_dict)
                all_cond_pooled.append([cond, pooled_dict])
                if show_pbar:
                    pbar.update(1)
                model_management.throw_exception_if_processing_interrupted()
            all_hooks.reset()
        return all_cond_pooled

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model()
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            self.add_hooks_to_dict(out)
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class VAE:
    # Conservative defaults for 12GB VRAM - 512px tiles
    DEFAULT_TILE_SIZE_LATENT = 64   # 64 latent = 512px for 8x compression
    DEFAULT_OVERLAP_LATENT = 8      # 8 latent = 64px overlap
    MIN_TILE_SIZE = 32

    def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None, no_init=False):
        if no_init:
            return
        if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys():
            sd = diffusers_convert.convert_vae_state_dict(sd)

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.latent_dim = 2
        self.output_channels = 3
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = False

        self.downscale_index_formula = None
        self.upscale_index_formula = None
        self.extra_1d_channel = None
        self.packed_latent_channels = None
        self.packed_latent_spatial_factor = 2
        self.latent_shift_factor = 0.0
        self.latent_scale_factor = 1.0
        self.latent_bn_running_mean = None
        self.latent_bn_running_var = None
        self.latent_bn_eps = 1.0e-5

        if config is None:
            if "decoder.mid.block_1.mix_factor" in sd:
                encoder_config = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                decoder_config = encoder_config.copy()
                decoder_config["video_kernel_size"] = [3, 1, 1]
                decoder_config["alpha"] = 0.0
                self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "ldm_patched.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                            encoder_config={'target': "ldm_patched.ldm.modules.diffusionmodules.model.Encoder", 'params': encoder_config},
                                                            decoder_config={'target': "ldm_patched.ldm.modules.temporal_ae.VideoDecoder", 'params': decoder_config})
            elif "taesd_decoder.1.weight" in sd:
                self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
                self.first_stage_model = ldm_patched.taesd.taesd.TAESD(latent_channels=self.latent_channels)
            elif "vquantizer.codebook.weight" in sd:
                self.first_stage_model = StageA()
                self.downscale_ratio = 4
                self.upscale_ratio = 4
                self.process_input = lambda image: image
                self.process_output = lambda image: image
            elif "backbone.1.0.block.0.1.num_batches_tracked" in sd:
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["encoder.{}".format(k)] = sd[k]
                sd = new_sd
            elif "blocks.11.num_batches_tracked" in sd:
                self.first_stage_model = StageC_coder()
                self.latent_channels = 16
                new_sd = {}
                for k in sd:
                    new_sd["previewer.{}".format(k)] = sd[k]
                sd = new_sd
            elif "encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd:
                self.first_stage_model = StageC_coder()
                self.downscale_ratio = 32
                self.latent_channels = 16
            elif "decoder.conv_in.weight" in sd:
                ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

                if 'encoder.down.2.downsample.conv.weight' not in sd and 'decoder.up.3.upsample.conv.weight' not in sd:
                    ddconfig['ch_mult'] = [1, 2, 4]
                    self.downscale_ratio = 4
                    self.upscale_ratio = 4

                self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.weight"].shape[1]
                if 'post_quant_conv.weight' in sd:
                    self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
                else:
                    self.first_stage_model = AutoencodingEngine(regularizer_config={'target': "ldm_patched.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
                                                                encoder_config={'target': "ldm_patched.ldm.modules.diffusionmodules.model.Encoder", 'params': ddconfig},
                                                                decoder_config={'target': "ldm_patched.ldm.modules.diffusionmodules.model.Decoder", 'params': ddconfig})
                if 'bn.running_mean' in sd and 'bn.running_var' in sd:
                    self.latent_bn_running_mean = sd.pop('bn.running_mean')
                    self.latent_bn_running_var = sd.pop('bn.running_var')
                    sd.pop('bn.num_batches_tracked', None)
                if self.latent_channels == 32:
                    self.latent_shift_factor = 0.0760
                    self.latent_scale_factor = 0.6043
            elif "decoder.layers.1.layers.0.beta" in sd:
                self.first_stage_model = AudioOobleckVAE()
                self.memory_used_encode = lambda shape, dtype: (1000 * shape[2]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * 2048) * model_management.dtype_size(dtype)
                self.latent_channels = 64
                self.output_channels = 2
                self.upscale_ratio = 2048
                self.downscale_ratio = 2048
                self.latent_dim = 1
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
                self.disable_offload = True
            elif "blocks.2.blocks.3.stack.5.weight" in sd or "decoder.blocks.2.blocks.3.stack.5.weight" in sd or "layers.4.layers.1.attn_block.attn.qkv.weight" in sd or "encoder.layers.4.layers.1.attn_block.attn.qkv.weight" in sd:
                if "blocks.2.blocks.3.stack.5.weight" in sd:
                    sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {"": "decoder."})
                if "layers.4.layers.1.attn_block.attn.qkv.weight" in sd:
                    sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {"": "encoder."})
                self.first_stage_model = ldm_patched.ldm.genmo.vae.model.VideoVAE()
                self.latent_channels = 12
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (1000 * shape[2] * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (1.5 * max(shape[2], 7) * shape[3] * shape[4] * (6 * 8 * 8)) * model_management.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 6 - 5), 8, 8)
                self.upscale_index_formula = (6, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 5) / 6)), 8, 8)
                self.downscale_index_formula = (6, 8, 8)
                self.working_dtypes = [torch.float16, torch.float32]
            elif "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd:
                tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
                version = 0
                if tensor_conv1.shape[0] == 512:
                    version = 0
                elif tensor_conv1.shape[0] == 1024:
                    version = 1
                    if "encoder.down_blocks.1.conv.conv.bias" in sd:
                        version = 2
                vae_config = None
                if metadata is not None and "config" in metadata:
                    vae_config = json.loads(metadata["config"]).get("vae", None)
                self.first_stage_model = ldm_patched.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE(version=version, config=vae_config)
                self.latent_channels = 128
                self.latent_dim = 3
                self.memory_used_decode = lambda shape, dtype: (900 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (70 * max(shape[2], 7) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)
                self.upscale_index_formula = (8, 32, 32)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 32, 32)
                self.downscale_index_formula = (8, 32, 32)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.conv_in.conv.weight" in sd:
                ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
                ddconfig["conv3d"] = True
                ddconfig["time_compress"] = 4
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = ddconfig['z_channels'] = sd["decoder.conv_in.conv.weight"].shape[1]
                self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd['post_quant_conv.weight'].shape[1])
                self.memory_used_decode = lambda shape, dtype: (1500 * shape[2] * shape[3] * shape[4] * (4 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (900 * max(shape[2], 2) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
            elif "decoder.unpatcher3d.wavelets" in sd:
                self.upscale_ratio = (lambda a: max(0, a * 8 - 7), 8, 8)
                self.upscale_index_formula = (8, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 7) / 8)), 8, 8)
                self.downscale_index_formula = (8, 8, 8)
                self.latent_dim = 3
                self.latent_channels = 16
                ddconfig = {'z_channels': 16, 'latent_channels': self.latent_channels, 'z_factor': 1, 'resolution': 1024, 'in_channels': 3, 'out_channels': 3, 'channels': 128, 'channels_mult': [2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [32], 'dropout': 0.0, 'patch_size': 4, 'num_groups': 1, 'temporal_compression': 8, 'spacial_compression': 8}
                self.first_stage_model = ldm_patched.ldm.cosmos.vae.CausalContinuousVideoTokenizer(**ddconfig)
                self.memory_used_decode = lambda shape, dtype: (50 * shape[2] * shape[3] * shape[4] * (8 * 8 * 8)) * model_management.dtype_size(dtype)
                self.memory_used_encode = lambda shape, dtype: (50 * (round((shape[2] + 7) / 8) * 8) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
                self.working_dtypes = [torch.bfloat16, torch.float32]
            elif "decoder.middle.0.residual.0.gamma" in sd:
                self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
                self.upscale_index_formula = (4, 8, 8)
                self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
                self.downscale_index_formula = (4, 8, 8)
                self.latent_dim = 3
                self.latent_channels = 16
                ddconfig = {"dim": 96, "z_dim": self.latent_channels, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_scales": [], "temperal_downsample": [False, True, True], "dropout": 0.0}
                self.first_stage_model = ldm_patched.ldm.wan.vae.WanVAE(**ddconfig)
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.memory_used_encode = lambda shape, dtype: 6000 * shape[3] * shape[4] * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: 7000 * shape[3] * shape[4] * (8 * 8) * model_management.dtype_size(dtype)
            elif "geo_decoder.cross_attn_decoder.ln_1.bias" in sd:
                self.latent_dim = 1
                ln_post = "geo_decoder.ln_post.weight" in sd
                inner_size = sd["geo_decoder.output_proj.weight"].shape[1]
                downsample_ratio = sd["post_kl.weight"].shape[0] // inner_size
                mlp_expand = sd["geo_decoder.cross_attn_decoder.mlp.c_fc.weight"].shape[0] // inner_size
                self.memory_used_encode = lambda shape, dtype: (1000 * shape[2]) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (1024 * 1024 * 1024 * 2.0) * model_management.dtype_size(dtype)
                ddconfig = {"embed_dim": 64, "num_freqs": 8, "include_pi": False, "heads": 16, "width": 1024, "num_decoder_layers": 16, "qkv_bias": False, "qk_norm": True, "geo_decoder_mlp_expand_ratio": mlp_expand, "geo_decoder_downsample_ratio": downsample_ratio, "geo_decoder_ln_post": ln_post}
                self.first_stage_model = ldm_patched.ldm.hunyuan3d.vae.ShapeVAE(**ddconfig)
                self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            elif "vocoder.backbone.channel_layers.0.0.bias" in sd:
                self.first_stage_model = ldm_patched.ldm.ace.vae.music_dcae_pipeline.MusicDCAE(source_sample_rate=44100)
                self.memory_used_encode = lambda shape, dtype: (shape[2] * 330) * model_management.dtype_size(dtype)
                self.memory_used_decode = lambda shape, dtype: (shape[2] * shape[3] * 87000) * model_management.dtype_size(dtype)
                self.latent_channels = 8
                self.output_channels = 2
                self.upscale_ratio = 4096
                self.downscale_ratio = 4096
                self.latent_dim = 2
                self.process_output = lambda audio: audio
                self.process_input = lambda audio: audio
                self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                self.disable_offload = True
                self.extra_1d_channel = 16
            else:
                logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']))
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        if device is None:
            device = model_management.vae_device()
        self.device = device
        offload_device = model_management.vae_offload_device()
        if dtype is None:
            dtype = model_management.vae_dtype(self.device, self.working_dtypes)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = model_management.intermediate_device()

        self.patcher = ldm_patched.modules.model_patcher.ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
        logging.info("VAE load device: {}, offload device: {}, dtype: {}".format(self.device, offload_device, self.vae_dtype))

        if shared.opts.reflective_padding_vae_sd == "Enabled":
            for module in self.first_stage_model.modules():
                from torch import nn
                logging.info(self)
                if isinstance(module, nn.Conv2d):
                    pad_h, pad_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                    if pad_h > 0 or pad_w > 0:
                        module.padding_mode = "reflect"
            logging.info("Setting reflective padding")

    def throw_exception_if_invalid(self):
        if self.first_stage_model is None:
            raise RuntimeError("ERROR: VAE is invalid: None\n\nIf the VAE is from a checkpoint loader node your checkpoint does not contain a valid VAE.")

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.upscale_ratio = self.upscale_ratio
        n.process_output = self.process_output
        n.output_channels = self.output_channels
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.process_input = self.process_input
        n.working_dtypes = self.working_dtypes
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.output_device = self.output_device
        n.latent_dim = self.latent_dim
        n.downscale_index_formula = self.downscale_index_formula
        n.upscale_index_formula = self.upscale_index_formula
        n.packed_latent_channels = self.packed_latent_channels
        n.packed_latent_spatial_factor = self.packed_latent_spatial_factor
        n.latent_shift_factor = self.latent_shift_factor
        n.latent_scale_factor = self.latent_scale_factor
        n.latent_bn_running_mean = self.latent_bn_running_mean
        n.latent_bn_running_var = self.latent_bn_running_var
        n.latent_bn_eps = self.latent_bn_eps
        return n

    def set_packed_latents(self, packed_channels, spatial_factor=2):
        self.packed_latent_channels = packed_channels
        self.packed_latent_spatial_factor = spatial_factor

    def set_latent_bn_stats(self, running_mean, running_var, eps: float = 1.0e-5):
        self.latent_bn_running_mean = running_mean
        self.latent_bn_running_var = running_var
        self.latent_bn_eps = eps

    def _apply_packed_latent_bn(self, latent, inverse: bool):
        mean = getattr(self, "latent_bn_running_mean", None)
        var = getattr(self, "latent_bn_running_var", None)
        if mean is None or var is None:
            return latent
        if latent.ndim < 4:
            return latent

        sf = 2
        bn_channels = int(mean.shape[0])
        if bn_channels % (sf ** 2) != 0:
            return latent
        packed_channels = bn_channels // (sf ** 2)
        if latent.shape[1] != packed_channels:
            return latent

        h0 = latent.shape[-2]
        w0 = latent.shape[-1]
        pad_h = (sf - (h0 % sf)) % sf
        pad_w = (sf - (w0 % sf)) % sf
        if pad_h != 0 or pad_w != 0:
            latent = torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h))

        h = latent.shape[-2]
        w = latent.shape[-1]
        packed = latent.reshape(latent.shape[0], packed_channels, h // sf, sf, w // sf, sf)
        packed = packed.permute(0, 1, 3, 5, 2, 4).reshape(latent.shape[0], bn_channels, h // sf, w // sf)

        mean_t = mean.to(device=packed.device, dtype=packed.dtype).view(1, bn_channels, 1, 1)
        var_t = var.to(device=packed.device, dtype=packed.dtype).view(1, bn_channels, 1, 1)
        denom = (var_t + float(getattr(self, "latent_bn_eps", 1.0e-5))) ** 0.5
        if inverse:
            packed = packed * denom + mean_t
        else:
            packed = (packed - mean_t) / denom

        unpacked = packed.reshape(packed.shape[0], packed_channels, sf, sf, packed.shape[-2], packed.shape[-1])
        unpacked = unpacked.permute(0, 1, 4, 2, 5, 3).reshape(packed.shape[0], packed_channels, packed.shape[-2] * sf, packed.shape[-1] * sf)
        return unpacked[..., :h0, :w0]

    def _to_vae_latent(self, latent):
        packed_channels = getattr(self, "packed_latent_channels", None)
        sf = getattr(self, "packed_latent_spatial_factor", 2)
        if self.latent_scale_factor != 1.0 or self.latent_shift_factor != 0.0:
            latent = (latent - self.latent_shift_factor) / self.latent_scale_factor
        if packed_channels is None and latent.shape[1] * (sf ** 2) == getattr(self, "latent_channels", latent.shape[1]):
            packed_channels = latent.shape[1]
        if packed_channels is not None and latent.shape[1] == packed_channels:
            if packed_channels * (sf ** 2) == self.latent_channels and latent.ndim >= 4:
                h = latent.shape[-2]
                w = latent.shape[-1]
                if h % sf != 0 or w % sf != 0:
                    pad_h = (sf - (h % sf)) % sf
                    pad_w = (sf - (w % sf)) % sf
                    latent = torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h))
                    h = latent.shape[-2]
                    w = latent.shape[-1]
                latent = latent.reshape(latent.shape[0], packed_channels, h // sf, sf, w // sf, sf)
                latent = latent.permute(0, 1, 3, 5, 2, 4).reshape(latent.shape[0], self.latent_channels, h // sf, w // sf)
        return latent

    def _from_vae_latent(self, latent):
        packed_channels = getattr(self, "packed_latent_channels", None)
        sf = getattr(self, "packed_latent_spatial_factor", 2)
        if packed_channels is None and self.latent_channels == 128 and latent.shape[1] == 128:
            packed_channels = 32
        if packed_channels is not None and latent.shape[1] == self.latent_channels:
            if packed_channels * (sf ** 2) == self.latent_channels and latent.ndim >= 4:
                h = latent.shape[-2]
                w = latent.shape[-1]
                latent = latent.reshape(latent.shape[0], packed_channels, sf, sf, h, w)
                latent = latent.permute(0, 1, 4, 2, 5, 3).reshape(latent.shape[0], packed_channels, h * sf, w * sf)
        if self.latent_scale_factor != 1.0 or self.latent_shift_factor != 0.0:
            latent = latent * self.latent_scale_factor + self.latent_shift_factor
        return latent

    def vae_encode_crop_pixels(self, pixels):
        downscale_ratio = self.spacial_compression_encode()

        dims = pixels.shape[1:-1]
        for d in range(len(dims)):
            x = (dims[d] // downscale_ratio) * downscale_ratio
            x_offset = (dims[d] % downscale_ratio) // 2
            if x != dims[d]:
                pixels = pixels.narrow(d + 1, x_offset, x)
        return pixels

    # ==================== OPTIMIZED TILING METHODS ====================

    def _decode_tile_to_cpu(self, samples):
        """Decode a single tile on GPU, return result on CPU immediately."""
        samples_gpu = self._to_vae_latent(samples).to(self.vae_dtype).to(self.device)
        decoded = self.first_stage_model.decode(samples_gpu)
        result = self.process_output(decoded.float()).cpu()
        return result

    def _encode_tile_to_cpu(self, pixels):
        """Encode a single tile on GPU, return result on CPU immediately."""
        pixels_gpu = self.process_input(pixels).to(self.vae_dtype).to(self.device)
        encoded = self.first_stage_model.encode(pixels_gpu)
        result = encoded.float().cpu()
        return result

    def _make_linear_ramp(self, size, direction='up'):
        """Create a 1D linear ramp on CPU for blending."""
        if direction == 'up':
            return torch.linspace(0.0, 1.0, size, dtype=torch.float32)
        else:
            return torch.linspace(1.0, 0.0, size, dtype=torch.float32)

    def _get_spatial_ratio(self):
        """Get the spatial compression ratio for 2D operations."""
        try:
            return self.upscale_ratio[-1] if isinstance(self.upscale_ratio, tuple) else self.upscale_ratio
        except:
            return self.upscale_ratio

    def _decode_tiled_cpu_accumulate(self, samples, tile_x, tile_y, overlap):
        """
        Tiled decode with CPU accumulation and linear blend ramps.
        GPU only processes one tile at a time, all blending on CPU.
        """
        b, c, h, w = samples.shape
        ratio = self._get_spatial_ratio()
        out_h = h * ratio
        out_w = w * ratio
        overlap_px = overlap * ratio

        # All accumulation on CPU (uses system RAM)
        output = torch.zeros((b, self.output_channels, out_h, out_w), dtype=torch.float32)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

        step_x = max(1, tile_x - overlap)
        step_y = max(1, tile_y - overlap)

        # Build tile list
        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x_end = min(x + tile_x, w)
                y_end = min(y + tile_y, h)
                tiles.append((x, y, x_end, y_end))
                x += step_x
            y += step_y

        pbar = ldm_patched.modules.utils.ProgressBar(len(tiles))

        for (x, y, x_end, y_end) in tiles:
            tile = samples[:, :, y:y_end, x:x_end]
            decoded = self._decode_tile_to_cpu(tile)

            ox = x * ratio
            oy = y * ratio
            ox_end = x_end * ratio
            oy_end = y_end * ratio

            th = oy_end - oy
            tw = ox_end - ox

            # Build blend mask on CPU
            blend = torch.ones((1, 1, th, tw), dtype=torch.float32)

            # Left edge blend
            if x > 0 and overlap_px > 0:
                ramp_len = min(overlap_px, tw)
                ramp = self._make_linear_ramp(ramp_len, 'up')
                blend[:, :, :, :ramp_len] *= ramp.view(1, 1, 1, -1)

            # Right edge blend
            if x_end < w and overlap_px > 0:
                ramp_len = min(overlap_px, tw)
                ramp = self._make_linear_ramp(ramp_len, 'down')
                blend[:, :, :, -ramp_len:] *= ramp.view(1, 1, 1, -1)

            # Top edge blend
            if y > 0 and overlap_px > 0:
                ramp_len = min(overlap_px, th)
                ramp = self._make_linear_ramp(ramp_len, 'up')
                blend[:, :, :ramp_len, :] *= ramp.view(1, 1, -1, 1)

            # Bottom edge blend
            if y_end < h and overlap_px > 0:
                ramp_len = min(overlap_px, th)
                ramp = self._make_linear_ramp(ramp_len, 'down')
                blend[:, :, -ramp_len:, :] *= ramp.view(1, 1, -1, 1)

            # Accumulate on CPU
            output[:, :, oy:oy_end, ox:ox_end] += decoded * blend
            weights[:, :, oy:oy_end, ox:ox_end] += blend

            pbar.update(1)

        return output / weights.clamp(min=1e-8)

    def _encode_tiled_cpu_accumulate(self, pixel_samples, tile_x, tile_y, overlap):
        """
        Tiled encode with CPU accumulation and linear blend ramps.
        GPU only processes one tile at a time, all blending on CPU.
        """
        b, c, h, w = pixel_samples.shape
        ratio = self._get_spatial_ratio()
        out_h = h // ratio
        out_w = w // ratio
        overlap_latent = overlap // ratio

        output = torch.zeros((b, self.latent_channels, out_h, out_w), dtype=torch.float32)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

        step_x = max(ratio, tile_x - overlap)
        step_y = max(ratio, tile_y - overlap)

        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x_end = min(x + tile_x, w)
                y_end = min(y + tile_y, h)
                tiles.append((x, y, x_end, y_end))
                x += step_x
            y += step_y

        pbar = ldm_patched.modules.utils.ProgressBar(len(tiles))

        for (x, y, x_end, y_end) in tiles:
            tile = pixel_samples[:, :, y:y_end, x:x_end]
            encoded = self._encode_tile_to_cpu(tile)

            ox = x // ratio
            oy = y // ratio
            ox_end = x_end // ratio
            oy_end = y_end // ratio

            th = oy_end - oy
            tw = ox_end - ox

            blend = torch.ones((1, 1, th, tw), dtype=torch.float32)

            if x > 0 and overlap_latent > 0:
                ramp_len = min(overlap_latent, tw)
                ramp = self._make_linear_ramp(ramp_len, 'up')
                blend[:, :, :, :ramp_len] *= ramp.view(1, 1, 1, -1)

            if x_end < w and overlap_latent > 0:
                ramp_len = min(overlap_latent, tw)
                ramp = self._make_linear_ramp(ramp_len, 'down')
                blend[:, :, :, -ramp_len:] *= ramp.view(1, 1, 1, -1)

            if y > 0 and overlap_latent > 0:
                ramp_len = min(overlap_latent, th)
                ramp = self._make_linear_ramp(ramp_len, 'up')
                blend[:, :, :ramp_len, :] *= ramp.view(1, 1, -1, 1)

            if y_end < h and overlap_latent > 0:
                ramp_len = min(overlap_latent, th)
                ramp = self._make_linear_ramp(ramp_len, 'down')
                blend[:, :, -ramp_len:, :] *= ramp.view(1, 1, -1, 1)

            output[:, :, oy:oy_end, ox:ox_end] += encoded * blend
            weights[:, :, oy:oy_end, ox:ox_end] += blend

            pbar.update(1)

        return output / weights.clamp(min=1e-8)

    def decode_tiled_(self, samples, tile_x=None, tile_y=None, overlap=None):
        """
        Tiled VAE decode optimized for 12GB VRAM.
        Uses 512px tiles (64 latent) with CPU accumulation.
        """
        b, c, h, w = samples.shape

        # Use conservative defaults (64 latent = 512px for 8x compression)
        if tile_x is None:
            tile_x = self.DEFAULT_TILE_SIZE_LATENT
        if tile_y is None:
            tile_y = self.DEFAULT_TILE_SIZE_LATENT
        if overlap is None:
            overlap = self.DEFAULT_OVERLAP_LATENT

        # Clamp to image size
        tile_x = min(tile_x, w)
        tile_y = min(tile_y, h)

        # Single tile case - no tiling needed
        if w <= tile_x and h <= tile_y:
            return self._decode_tile_to_cpu(samples).to(self.output_device)

        # Clear GPU memory once before starting (not between tiles!)
        if self.device.type != 'cpu':
            torch.cuda.empty_cache()

        result = self._decode_tiled_cpu_accumulate(samples, tile_x, tile_y, overlap)
        return result.to(self.output_device)

    def encode_tiled_(self, pixel_samples, tile_x=None, tile_y=None, overlap=None):
        """
        Tiled VAE encode optimized for 12GB VRAM.
        Uses 512px tiles with CPU accumulation.
        """
        b, c, h, w = pixel_samples.shape
        ratio = self._get_spatial_ratio()

        # Defaults in pixel space (512px tiles)
        if tile_x is None:
            tile_x = self.DEFAULT_TILE_SIZE_LATENT * ratio
        if tile_y is None:
            tile_y = self.DEFAULT_TILE_SIZE_LATENT * ratio
        if overlap is None:
            overlap = self.DEFAULT_OVERLAP_LATENT * ratio

        tile_x = min(tile_x, w)
        tile_y = min(tile_y, h)

        if w <= tile_x and h <= tile_y:
            return self._from_vae_latent(self._encode_tile_to_cpu(pixel_samples)).to(self.output_device)

        if self.device.type != 'cpu':
            torch.cuda.empty_cache()

        result = self._encode_tiled_cpu_accumulate(pixel_samples, tile_x, tile_y, overlap)
        return self._from_vae_latent(result).to(self.output_device)

    # Keep original methods for 1D/3D that use ldm_patched utilities
    def decode_tiled_1d(self, samples, tile_x=128, overlap=32):
        if samples.ndim == 3:
            decode_fn = lambda a: self.first_stage_model.decode(self._to_vae_latent(a).to(self.vae_dtype).to(self.device)).float()
        else:
            og_shape = samples.shape
            samples = samples.reshape((og_shape[0], og_shape[1] * og_shape[2], -1))
            decode_fn = lambda a: self.first_stage_model.decode(self._to_vae_latent(a.reshape((-1, og_shape[1], og_shape[2], a.shape[-1]))).to(self.vae_dtype).to(self.device)).float()

        return self.process_output(ldm_patched.modules.utils.tiled_scale_multidim(samples, decode_fn, tile=(tile_x,), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, output_device=self.output_device))

    def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
        decode_fn = lambda a: self.first_stage_model.decode(self._to_vae_latent(a).to(self.vae_dtype).to(self.device)).float()
        input_samples = self._to_vae_latent(samples)
        return self.process_output(ldm_patched.modules.utils.tiled_scale_multidim(input_samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, index_formulas=self.upscale_index_formula, output_device=self.output_device))

    def encode_tiled_1d(self, samples, tile_x=256 * 2048, overlap=64 * 2048):
        if self.latent_dim == 1:
            encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).float()
            out_channels = self.latent_channels
            upscale_amount = 1 / self.downscale_ratio
        else:
            extra_channel_size = self.extra_1d_channel
            out_channels = self.latent_channels * extra_channel_size
            tile_x = tile_x // extra_channel_size
            overlap = overlap // extra_channel_size
            upscale_amount = 1 / self.downscale_ratio
            encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).reshape(1, out_channels, -1).float()

        out = ldm_patched.modules.utils.tiled_scale_multidim(samples, encode_fn, tile=(tile_x,), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=self.output_device)
        out = self._from_vae_latent(out)
        if self.latent_dim == 1:
            return out
        else:
            return out.reshape(samples.shape[0], self.latent_channels, extra_channel_size, -1)

    def encode_tiled_3d(self, pixel_samples, tile_t=9999, tile_x=256, tile_y=256, overlap=(1, 64, 64)):
        encode_fn = lambda a: self.first_stage_model.encode(self.process_input(a).to(self.vae_dtype).to(self.device)).float()

        if isinstance(self.downscale_ratio, tuple):
            upscale_amount = (self.downscale_ratio[0], 1/self.downscale_ratio[1], 1/self.downscale_ratio[2])
        else:
            upscale_amount = 1/self.downscale_ratio

        samples = ldm_patched.modules.utils.tiled_scale_multidim(
            pixel_samples,
            encode_fn,
            tile=(tile_t, tile_x, tile_y),
            overlap=overlap,
            upscale_amount=upscale_amount,
            out_channels=self.latent_channels,
            index_formulas=self.downscale_index_formula,
            output_device=self.output_device
        )
        return self._from_vae_latent(samples)

    def decode_inner(self, samples_in):
        """Always use tiled decode for memory safety."""
        model_management.load_models_gpu([self.patcher], force_full_load=self.disable_offload)
        return self.decode_tiled_(samples_in).to(self.output_device)

    def decode(self, samples_in, vae_options={}):
        self.throw_exception_if_invalid()
        model_management.load_models_gpu([self.patcher], force_full_load=self.disable_offload)

        dims = samples_in.ndim - 2

        if dims == 1 or self.extra_1d_channel is not None:
            pixel_samples = self.decode_tiled_1d(samples_in)
        elif dims == 2:
            pixel_samples = self.decode_tiled_(samples_in)
        elif dims == 3:
            tile = 256 // self.spacial_compression_decode()
            overlap = tile // 4
            pixel_samples = self.decode_tiled_3d(samples_in, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))
        else:
            pixel_samples = self.decode_tiled_(samples_in)

        return pixel_samples.to(self.output_device).movedim(1, -1)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        self.throw_exception_if_invalid()
        model_management.load_models_gpu([self.patcher], force_full_load=self.disable_offload)

        dims = samples.ndim - 2
        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        if dims == 1:
            args.pop("tile_y", None)
            output = self.decode_tiled_1d(samples, **args)
        elif dims == 2:
            output = self.decode_tiled_(samples, **args)
        elif dims == 3:
            if overlap_t is None:
                args["overlap"] = (1, args.get("overlap", self.DEFAULT_OVERLAP_LATENT), args.get("overlap", self.DEFAULT_OVERLAP_LATENT))
            else:
                args["overlap"] = (max(1, overlap_t), args.get("overlap", self.DEFAULT_OVERLAP_LATENT), args.get("overlap", self.DEFAULT_OVERLAP_LATENT))
            if tile_t is not None:
                args["tile_t"] = max(2, tile_t)
            output = self.decode_tiled_3d(samples, **args)
        else:
            output = self.decode_tiled_(samples, **args)

        return output.movedim(1, -1)

    def encode_inner(self, pixel_samples):
        """Always use tiled encode for memory safety."""
        model_management.load_models_gpu([self.patcher], force_full_load=self.disable_offload)
        pixel_samples = pixel_samples.movedim(-1, 1)
        return self.encode_tiled_(pixel_samples)

    def encode(self, pixel_samples):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)

        if self.latent_dim == 3 and pixel_samples.ndim < 5:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)

        model_management.load_models_gpu([self.patcher], force_full_load=self.disable_offload)

        if self.latent_dim == 1 or self.extra_1d_channel is not None:
            samples = self.encode_tiled_1d(pixel_samples)
        elif self.latent_dim == 3:
            tile = 256
            overlap = tile // 4
            samples = self.encode_tiled_3d(pixel_samples, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))
        else:
            samples = self.encode_tiled_(pixel_samples)

        return samples

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        dims = self.latent_dim
        pixel_samples = pixel_samples.movedim(-1, 1)
        if dims == 3:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)

        model_management.load_models_gpu([self.patcher], force_full_load=self.disable_offload)

        args = {}
        if tile_x is not None:
            args["tile_x"] = tile_x
        if tile_y is not None:
            args["tile_y"] = tile_y
        if overlap is not None:
            args["overlap"] = overlap

        if dims == 1:
            args.pop("tile_y", None)
            samples = self.encode_tiled_1d(pixel_samples, **args)
        elif dims == 2:
            samples = self.encode_tiled_(pixel_samples, **args)
        elif dims == 3:
            if tile_t is not None:
                tile_t_latent = max(2, self.downscale_ratio[0](tile_t))
            else:
                tile_t_latent = 9999
            args["tile_t"] = self.upscale_ratio[0](tile_t_latent) if callable(self.upscale_ratio[0]) else tile_t_latent

            if overlap_t is None:
                args["overlap"] = (1, args.get("overlap", 64), args.get("overlap", 64))
            else:
                if callable(self.upscale_ratio[0]) and callable(self.downscale_ratio[0]):
                    args["overlap"] = (self.upscale_ratio[0](max(1, min(tile_t_latent // 2, self.downscale_ratio[0](overlap_t)))), args.get("overlap", 64), args.get("overlap", 64))
                else:
                    args["overlap"] = (max(1, overlap_t), args.get("overlap", 64), args.get("overlap", 64))

            maximum = pixel_samples.shape[2]
            if callable(self.upscale_ratio[0]) and callable(self.downscale_ratio[0]):
                maximum = self.upscale_ratio[0](self.downscale_ratio[0](maximum))

            samples = self.encode_tiled_3d(pixel_samples[:, :, :maximum], **args)
        else:
            samples = self.encode_tiled_(pixel_samples, **args)

        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()

    def spacial_compression_decode(self):
        try:
            return self.upscale_ratio[-1]
        except:
            return self.upscale_ratio

    def spacial_compression_encode(self):
        try:
            return self.downscale_ratio[-1]
        except:
            return self.downscale_ratio

    def temporal_compression_decode(self):
        try:
            return round(self.upscale_ratio[0](8192) / 8192)
        except:
            return None


class StyleModel:
    def __init__(self, model, device="cpu"):
        self.model = model

    def get_cond(self, input):
        return self.model(input.last_hidden_state)


def load_style_model(ckpt_path):
    model_data = ldm_patched.modules.utils.load_torch_file(ckpt_path, safe_load=True)
    keys = model_data.keys()
    if "style_embedding" in keys:
        model = ldm_patched.t2ia.adapter.StyleAdapter(width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8)
    elif "redux_down.weight" in keys:
        model = ldm_patched.ldm.flux.redux.ReduxImageEncoder()
    else:
        raise Exception("invalid style model {}".format(ckpt_path))
    model.load_state_dict(model_data)
    return StyleModel(model)

class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6
    MOCHI = 7
    LTXV = 8
    HUNYUAN_VIDEO = 9
    PIXART = 10
    COSMOS = 11
    LUMINA2 = 12
    WAN = 13
    HIDREAM = 14
    CHROMA = 15
    ACE = 16

def load_clip(ckpt_paths, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}):
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(ldm_patched.modules.utils.load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options)


class TEModel(Enum):
    CLIP_L = 1
    CLIP_H = 2
    CLIP_G = 3
    T5_XXL = 4
    T5_XL = 5
    T5_BASE = 6
    LLAMA3_8 = 7
    T5_XXL_OLD = 8
    GEMMA_2_2B = 9
    QWEN25_3B = 10

def detect_te_model(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TEModel.CLIP_G
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TEModel.CLIP_H
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TEModel.CLIP_L
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            return TEModel.T5_XXL
        elif weight.shape[-1] == 2048:
            return TEModel.T5_XL
    if 'encoder.block.23.layer.1.DenseReluDense.wi.weight' in sd:
        return TEModel.T5_XXL_OLD
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        return TEModel.T5_BASE
    if 'model.layers.0.post_feedforward_layernorm.weight' in sd:
        return TEModel.GEMMA_2_2B
    if 'model.layers.0.self_attn.k_proj.bias' in sd:
        return TEModel.QWEN25_3B
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        return TEModel.LLAMA3_8
    return None

def t5xxl_detect(clip_data):
    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    weight_name_old = "encoder.block.23.layer.1.DenseReluDense.wi.weight"

    for sd in clip_data:
        if weight_name in sd or weight_name_old in sd:
            return ldm_patched.modules.text_encoders.sd3_clip.t5_xxl_detect(sd)

    return {}

def llama_detect(clip_data):
    weight_name = "model.layers.0.self_attn.k_proj.weight"

    for sd in clip_data:
        if weight_name in sd:
            return ldm_patched.modules.text_encoders.hunyuan_video.llama_detect(sd)

    return {}

def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = ldm_patched.modules.utils.clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1)

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 1:
        te_model = detect_te_model(clip_data[0])
        if te_model == TEModel.CLIP_G:
            if clip_type == CLIPType.STABLE_CASCADE:
                clip_target.clip = sdxl_clip.StableCascadeClipModel
                clip_target.tokenizer = sdxl_clip.StableCascadeTokenizer
            elif clip_type == CLIPType.SD3:
                clip_target.clip = ldm_patched.modules.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=True, t5=False)
                clip_target.tokenizer = ldm_patched.modules.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = ldm_patched.modules.text_encoders.hidream.hidream_clip(clip_l=False, clip_g=True, t5=False, llama=False, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None)
                clip_target.tokenizer = ldm_patched.modules.text_encoders.hidream.HiDreamTokenizer
            else:
                clip_target.clip = sdxl_clip.SDXLRefinerClipModel
                clip_target.tokenizer = sdxl_clip.SDXLTokenizer
        elif te_model == TEModel.CLIP_H:
            clip_target.clip = ldm_patched.modules.text_encoders.sd2_clip.SD2ClipModel
            clip_target.tokenizer = ldm_patched.modules.text_encoders.sd2_clip.SD2Tokenizer
        elif te_model == TEModel.T5_XXL:
            if clip_type == CLIPType.SD3:
                clip_target.clip = ldm_patched.modules.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=False, t5=True, **t5xxl_detect(clip_data))
                clip_target.tokenizer = ldm_patched.modules.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.LTXV:
                clip_target.clip = ldm_patched.modules.text_encoders.lt.ltxv_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = ldm_patched.modules.text_encoders.lt.LTXVT5Tokenizer
            elif clip_type == CLIPType.PIXART or clip_type == CLIPType.CHROMA:
                clip_target.clip = ldm_patched.modules.text_encoders.pixart_t5.pixart_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = ldm_patched.modules.text_encoders.pixart_t5.PixArtTokenizer
            elif clip_type == CLIPType.WAN:
                clip_target.clip = ldm_patched.modules.text_encoders.wan.te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = ldm_patched.modules.text_encoders.wan.WanT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = ldm_patched.modules.text_encoders.hidream.hidream_clip(**t5xxl_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=True, llama=False, dtype_llama=None, llama_scaled_fp8=None)
                clip_target.tokenizer = ldm_patched.modules.text_encoders.hidream.HiDreamTokenizer
            else:
                clip_target.clip = ldm_patched.modules.text_encoders.genmo.mochi_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = ldm_patched.modules.text_encoders.genmo.MochiT5Tokenizer
        elif te_model == TEModel.T5_XXL_OLD:
            clip_target.clip = ldm_patched.modules.text_encoders.cosmos.te(**t5xxl_detect(clip_data))
            clip_target.tokenizer = ldm_patched.modules.text_encoders.cosmos.CosmosT5Tokenizer
        elif te_model == TEModel.T5_XL:
            clip_target.clip = ldm_patched.modules.text_encoders.aura_t5.AuraT5Model
            clip_target.tokenizer = ldm_patched.modules.text_encoders.aura_t5.AuraT5Tokenizer
        elif te_model == TEModel.T5_BASE:
            if clip_type == CLIPType.ACE or "spiece_model" in clip_data[0]:
                clip_target.clip = ldm_patched.modules.text_encoders.ace.AceT5Model
                clip_target.tokenizer = ldm_patched.modules.text_encoders.ace.AceT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            else:
                clip_target.clip = ldm_patched.modules.text_encoders.sa_t5.SAT5Model
                clip_target.tokenizer = ldm_patched.modules.text_encoders.sa_t5.SAT5Tokenizer
        elif te_model == TEModel.GEMMA_2_2B:
            clip_target.clip = ldm_patched.modules.text_encoders.lumina2.te(**llama_detect(clip_data))
            clip_target.tokenizer = ldm_patched.modules.text_encoders.lumina2.LuminaTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.LLAMA3_8:
            clip_target.clip = ldm_patched.modules.text_encoders.hidream.hidream_clip(**llama_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=False, llama=True, dtype_t5=None, t5xxl_scaled_fp8=None)
            clip_target.tokenizer = ldm_patched.modules.text_encoders.hidream.HiDreamTokenizer
        elif te_model == TEModel.QWEN25_3B:
            clip_target.clip = ldm_patched.modules.text_encoders.omnigen2.te(**llama_detect(clip_data))
            clip_target.tokenizer = ldm_patched.modules.text_encoders.omnigen2.Omnigen2Tokenizer
        else:
            if clip_type == CLIPType.SD3:
                clip_target.clip = ldm_patched.modules.text_encoders.sd3_clip.sd3_clip(clip_l=True, clip_g=False, t5=False)
                clip_target.tokenizer = ldm_patched.modules.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = ldm_patched.modules.text_encoders.hidream.hidream_clip(clip_l=True, clip_g=False, t5=False, llama=False, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None)
                clip_target.tokenizer = ldm_patched.modules.text_encoders.hidream.HiDreamTokenizer
            else:
                clip_target.clip = sd1_clip.SD1ClipModel
                clip_target.tokenizer = sd1_clip.SD1Tokenizer
    elif len(clip_data) == 2:
        if clip_type == CLIPType.SD3:
            te_models = [detect_te_model(clip_data[0]), detect_te_model(clip_data[1])]
            clip_target.clip = ldm_patched.modules.text_encoders.sd3_clip.sd3_clip(clip_l=TEModel.CLIP_L in te_models, clip_g=TEModel.CLIP_G in te_models, t5=TEModel.T5_XXL in te_models, **t5xxl_detect(clip_data))
            clip_target.tokenizer = ldm_patched.modules.text_encoders.sd3_clip.SD3Tokenizer
        elif clip_type == CLIPType.HUNYUAN_DIT:
            clip_target.clip = ldm_patched.modules.text_encoders.hydit.HyditModel
            clip_target.tokenizer = ldm_patched.modules.text_encoders.hydit.HyditTokenizer
        elif clip_type == CLIPType.FLUX:
            clip_target.clip = ldm_patched.modules.text_encoders.flux.flux_clip(**t5xxl_detect(clip_data))
            clip_target.tokenizer = ldm_patched.modules.text_encoders.flux.FluxTokenizer
        elif clip_type == CLIPType.HUNYUAN_VIDEO:
            clip_target.clip = ldm_patched.modules.text_encoders.hunyuan_video.hunyuan_video_clip(**llama_detect(clip_data))
            clip_target.tokenizer = ldm_patched.modules.text_encoders.hunyuan_video.HunyuanVideoTokenizer
        elif clip_type == CLIPType.HIDREAM:
            hidream_dualclip_classes = []
            for hidream_te in clip_data:
                te_model = detect_te_model(hidream_te)
                hidream_dualclip_classes.append(te_model)

            clip_l = TEModel.CLIP_L in hidream_dualclip_classes
            clip_g = TEModel.CLIP_G in hidream_dualclip_classes
            t5 = TEModel.T5_XXL in hidream_dualclip_classes
            llama = TEModel.LLAMA3_8 in hidream_dualclip_classes

            t5_kwargs = t5xxl_detect(clip_data) if t5 else {}
            llama_kwargs = llama_detect(clip_data) if llama else {}

            clip_target.clip = ldm_patched.modules.text_encoders.hidream.hidream_clip(clip_l=clip_l, clip_g=clip_g, t5=t5, llama=llama, **t5_kwargs, **llama_kwargs)
            clip_target.tokenizer = ldm_patched.modules.text_encoders.hidream.HiDreamTokenizer
        else:
            clip_target.clip = sdxl_clip.SDXLClipModel
            clip_target.tokenizer = sdxl_clip.SDXLTokenizer
    elif len(clip_data) == 3:
        clip_target.clip = ldm_patched.modules.text_encoders.sd3_clip.sd3_clip(**t5xxl_detect(clip_data))
        clip_target.tokenizer = ldm_patched.modules.text_encoders.sd3_clip.SD3Tokenizer
    elif len(clip_data) == 4:
        clip_target.clip = ldm_patched.modules.text_encoders.hidream.hidream_clip(**t5xxl_detect(clip_data), **llama_detect(clip_data))
        clip_target.tokenizer = ldm_patched.modules.text_encoders.hidream.HiDreamTokenizer

    parameters = 0
    for c in clip_data:
        parameters += ldm_patched.modules.utils.calculate_parameters(c)
        tokenizer_data, model_options = ldm_patched.modules.text_encoders.long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    clip = CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip

def load_gligen(ckpt_path):
    data = ldm_patched.modules.utils.load_torch_file(ckpt_path, safe_load=True)
    model = gligen.load_gligen(data)
    if model_management.should_use_fp16():
        model = model.half()
    return ldm_patched.modules.model_patcher.ModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=model_management.unet_offload_device())

def model_detection_error_hint(path, state_dict):
    filename = os.path.basename(path)
    if 'lora' in filename.lower():
        return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded via <lora:loraname:lorastrength>..."
    return ""

def load_checkpoint(config_path=None, ckpt_path=None, output_vae=True, output_clip=True, embedding_directory=None, state_dict=None, config=None):
    logging.warning("Warning: The load checkpoint with config function is deprecated and will eventually be removed, please use the other one.")
    model, clip, vae, _ = load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=output_clip, output_clipvision=False, embedding_directory=embedding_directory, output_model=True)
    if config is None:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
    model_config_params = config['model']['params']
    clip_config = model_config_params['cond_stage_config']

    if "parameterization" in model_config_params:
        if model_config_params["parameterization"] == "v":
            m = model.clone()
            class ModelSamplingAdvanced(ldm_patched.modules.model_sampling.ModelSamplingDiscrete, ldm_patched.modules.model_sampling.V_PREDICTION):
                pass
            m.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config))
            model = m

    layer_idx = clip_config.get("params", {}).get("layer_idx", None)
    if layer_idx is not None:
        clip.clip_layer(layer_idx)

    return (model, clip, vae)

def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
    sd, metadata = ldm_patched.modules.utils.load_torch_file(ckpt_path, return_metadata=True)
    out = load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options=te_model_options, metadata=metadata)
    if out is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(ckpt_path, model_detection_error_hint(ckpt_path, sd)))
    return out

def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}, metadata=None):
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    parameters = ldm_patched.modules.utils.calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype = ldm_patched.modules.utils.weight_dtype(sd, diffusion_model_prefix)
    load_device = model_management.get_torch_device()

    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
    if model_config is None:
        logging.warning("Warning, This is not a checkpoint file, trying to load it as a diffusion model only.")
        diffusion_model = load_diffusion_model_state_dict(sd, model_options={})
        if diffusion_model is None:
            return None
        return (diffusion_model, None, VAE(sd={}), None)

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

    if unet_dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    if output_model:
        inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        model.load_model_weights(sd, diffusion_model_prefix)

    if output_vae:
        vae_sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd, metadata=metadata)
        if hasattr(model_config, "packed_vae_latent_channels"):
            vae.set_packed_latents(model_config.packed_vae_latent_channels, getattr(model_config, "packed_vae_spatial_factor", 2))

    if output_clip:
        clip_target = model_config.clip_target(state_dict=sd)
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                parameters = ldm_patched.modules.utils.calculate_parameters(clip_sd)
                clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options=te_model_options)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = ldm_patched.modules.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded diffusion model directly to GPU")
            model_management.load_models_gpu([model_patcher], force_full_load=True)

    return (model_patcher, clip, vae, clipvision)


def load_diffusion_model_state_dict(sd, model_options={}):
    dtype = model_options.get("dtype", None)

    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = ldm_patched.modules.utils.calculate_parameters(sd)
    weight_dtype = ldm_patched.modules.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = ldm_patched.modules.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    return ldm_patched.modules.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

def load_diffusion_model(unet_path, model_options={}):
    sd = ldm_patched.modules.utils.load_torch_file(unet_path)
    model = load_diffusion_model_state_dict(sd, model_options=model_options)
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model

def load_unet(unet_path, dtype=None):
    print("WARNING: the load_unet function has been deprecated and will be removed please switch to: load_diffusion_model")
    return load_diffusion_model(unet_path, model_options={"dtype": dtype})

def load_unet_state_dict(sd, dtype=None):
    print("WARNING: the load_unet_state_dict function has been deprecated and will be removed please switch to: load_diffusion_model_state_dict")
    return load_diffusion_model_state_dict(sd, model_options={"dtype": dtype})


def save_checkpoint(output_path, model, clip=None, vae=None, clip_vision=None, metadata=None, extra_keys={}):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()
    vae_sd = None
    if vae is not None:
        vae_sd = vae.get_sd()

    model_management.load_models_gpu(load_models, force_patch_weights=True)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    sd = model.model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    for k in sd:
        t = sd[k]
        if not t.is_contiguous():
            sd[k] = t.contiguous()

    ldm_patched.modules.utils.save_torch_file(sd, output_path, metadata=metadata)
