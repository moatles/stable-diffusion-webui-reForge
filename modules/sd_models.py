import collections
import importlib
import os
import sys
import threading
import enum
import gc
import weakref
import logging as log
from typing import Optional, Dict, Any, Set
from urllib import request

import torch
import re
import safetensors.torch
from omegaconf import OmegaConf, ListConfig
import numpy as np
import psutil  # Bug fix: Was used but not imported

import ldm.modules.midas as midas
from modules import (
    paths, shared, modelloader, devices, script_callbacks,
    sd_vae, errors, hashes, sd_models_config, cache,
    extra_networks, processing, patches
)
from modules.timer import Timer
from modules_forge import forge_loader
import ldm_patched.modules.utils
from ldm_patched.modules import model_management
from ldm_patched.modules.patcher_extension import PatcherInjection

# ============================================================================
# Constants and Global State
# ============================================================================

MODEL_DIR = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, MODEL_DIR))

checkpoints_list: Dict[str, 'CheckpointInfo'] = {}
checkpoint_aliases: Dict[str, 'CheckpointInfo'] = {}
checkpoint_alisases = checkpoint_aliases  # Backward compatibility alias
checkpoints_loaded: collections.OrderedDict = collections.OrderedDict()

# Memory tracking
_models_loaded_count = 0
_peak_memory_usage = 0
_disable_checkpoint_caching = True


class ModelType(enum.Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SSD = 4
    SD3 = 5


# ============================================================================
# VAE Structure Preservation
# ============================================================================

class VAEStructurePreserver(PatcherInjection):
    """Injection that preserves VAE structure during model unloading."""

    def __init__(self):
        self.preserved_vae_refs: Set[int] = set()

    def inject(self, model_patcher) -> None:
        self.preserved_vae_refs.clear()
        model = model_patcher.model

        if not hasattr(model, 'forge_objects'):
            return

        forge_objects = model.forge_objects
        if not hasattr(forge_objects, 'vae'):
            return

        vae = forge_objects.vae
        self._add_ref(vae)

        if hasattr(vae, 'model'):
            vae_model = vae.model
            self._add_ref(vae_model)

            for attr in ('encoder', 'decoder', 'quantize'):
                if hasattr(vae_model, attr):
                    self._add_ref(getattr(vae_model, attr))

        if hasattr(vae, 'patcher'):
            self._add_ref(vae.patcher)

    def _add_ref(self, obj: Any) -> None:
        if obj is not None:
            self.preserved_vae_refs.add(id(obj))

    def eject(self, model_patcher) -> None:
        self.preserved_vae_refs.clear()


# ============================================================================
# Safe Attribute Setting (Patched Functions)
# ============================================================================

_original_set_attr = ldm_patched.modules.utils.set_attr
_original_set_attr_param = ldm_patched.modules.utils.set_attr_param


def safer_set_attr(obj: Any, attr: str, value: Any) -> Optional[Any]:
    """Safer version of set_attr with None checks and error handling."""
    try:
        attrs = attr.split(".")
        for name in attrs[:-1]:
            if obj is None or not hasattr(obj, name):
                return None
            obj = getattr(obj, name)

        if obj is None:
            return None

        prev = getattr(obj, attrs[-1], None)
        setattr(obj, attrs[-1], value)
        return prev
    except Exception as e:
        log.debug(f"Error in safer_set_attr for {attr}: {e}")
        return None


def safer_set_attr_param(obj: Any, attr: str, value: Any) -> Optional[Any]:
    """Safer version of set_attr_param with None handling."""
    if value is None:
        return None
    try:
        return safer_set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))
    except Exception as e:
        log.debug(f"Error in safer_set_attr_param for {attr}: {e}")
        return None


# Apply patches
ldm_patched.modules.utils.set_attr = safer_set_attr
ldm_patched.modules.utils.set_attr_param = safer_set_attr_param


# ============================================================================
# VAE Validation
# ============================================================================

def validate_and_fix_vae(sd_model) -> None:
    """Check and attempt to fix VAE if components are missing."""
    if not hasattr(sd_model, 'forge_objects'):
        return

    if not hasattr(sd_model.forge_objects, 'vae'):
        log.warning("Model has no VAE object")
        return

    vae = sd_model.forge_objects.vae

    if not hasattr(vae, 'model'):
        _reload_vae(sd_model)
        return

    missing = []
    for component in ('encoder', 'decoder'):
        if not hasattr(vae.model, component) or getattr(vae.model, component) is None:
            missing.append(component)

    if missing:
        log.warning(f"VAE missing components: {', '.join(missing)}, reloading")
        _reload_vae(sd_model)


def _reload_vae(sd_model) -> None:
    """Helper to reload VAE."""
    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(sd_model.sd_checkpoint_info.filename).tuple()
    sd_vae.load_vae(sd_model, vae_file, vae_source)


# ============================================================================
# Checkpoint Info
# ============================================================================

def replace_key(d: dict, key: str, new_key: str, value: Any) -> dict:
    """Replace a key in dict while preserving order."""
    keys = list(d.keys())
    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}
    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename: str):
        self.filename = filename
        abspath = os.path.abspath(filename)
        abs_ckpt_dir = os.path.abspath(shared.cmd_opts.ckpt_dir) if shared.cmd_opts.ckpt_dir else None

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        # Determine name based on path
        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        name = name.lstrip("\\/")

        self.metadata = {}
        self.modelspec_thumbnail = None

        if self.is_safetensors:
            self._load_safetensors_metadata(name)

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)
        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
        self.shorthash = self.sha256[:10] if self.sha256 else None

        self.title = f'{name} [{self.shorthash}]' if self.shorthash else name
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]' if self.shorthash else self.name_for_extra

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids.extend([self.shorthash, self.sha256,
                           f'{self.name} [{self.shorthash}]',
                           f'{self.name_for_extra} [{self.shorthash}]'])

    def _load_safetensors_metadata(self, name: str) -> None:
        """Load metadata from safetensors file."""
        def read_metadata():
            metadata = read_metadata_from_safetensors(self.filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)
            return metadata

        try:
            self.metadata = cache.cached_data_for_file(
                'safetensors-metadata', f"checkpoint/{name}", self.filename, read_metadata
            )
        except Exception as e:
            errors.display(e, f"reading metadata for {self.filename}")

    def register(self) -> None:
        checkpoints_list[self.title] = self
        for id_ in self.ids:
            checkpoint_aliases[id_] = self

    def calculate_shorthash(self) -> Optional[str]:
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return None

        shorthash = self.sha256[:10]
        if self.shorthash == shorthash:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids.extend([self.shorthash, self.sha256,
                           f'{self.name} [{self.shorthash}]',
                           f'{self.name_for_extra} [{self.shorthash}]'])

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash


# ============================================================================
# Model Setup and Listing
# ============================================================================

try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    pass


def setup_model() -> None:
    """One-time setup tasks for SD models."""
    os.makedirs(model_path, exist_ok=True)
    enable_midas_autodownload()
    patch_given_betas()


def checkpoint_tiles(use_short: bool = False) -> list:
    return [x.short_title if use_short else x.title for x in checkpoints_list.values()]


def list_models() -> None:
    checkpoints_list.clear()
    checkpoint_aliases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
    else:
        model_url = "https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors"

    model_list = modelloader.load_models(
        model_path=model_path,
        model_url=model_url,
        command_path=shared.cmd_opts.ckpt_dir,
        ext_filter=[".ckpt", ".safetensors"],
        download_name="NoobAI-XL-v1.1.safetensors",
        ext_blacklist=[".vae.ckpt", ".vae.safetensors"]
    )

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()
        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found: {cmd_ckpt}", file=sys.stderr)

    for filename in model_list:
        CheckpointInfo(filename).register()


re_strip_checksum = re.compile(r"\s*\[[^]]+]\s*$")


def get_closet_checkpoint_match(search_string: str) -> Optional[CheckpointInfo]:
    if not search_string:
        return None

    checkpoint_info = checkpoint_aliases.get(search_string)
    if checkpoint_info:
        return checkpoint_info

    # Search by title
    found = sorted(
        [info for info in checkpoints_list.values() if search_string in info.title],
        key=lambda x: len(x.title)
    )
    if found:
        return found[0]

    # Search without checksum
    search_string_without_checksum = re.sub(re_strip_checksum, '', search_string)
    found = sorted(
        [info for info in checkpoints_list.values() if search_string_without_checksum in info.title],
        key=lambda x: len(x.title)
    )
    return found[0] if found else None


def model_hash(filename: str) -> str:
    """Calculate a quick (collision-prone) hash of model file."""
    try:
        import hashlib
        with open(filename, "rb") as file:
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint() -> CheckpointInfo:
    """Select a checkpoint. Raises FileNotFoundError if none found."""
    model_checkpoint = shared.opts.sd_model_checkpoint
    checkpoint_info = checkpoint_aliases.get(model_checkpoint)

    if checkpoint_info:
        return checkpoint_info

    if not checkpoints_list:
        error_message = "No checkpoints found. Searched:\n"
        if shared.cmd_opts.ckpt:
            error_message += f" - file {os.path.abspath(shared.cmd_opts.ckpt)}\n"
        error_message += f" - directory {model_path}\n"
        if shared.cmd_opts.ckpt_dir:
            error_message += f" - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}\n"
        error_message += "Find and place a .ckpt or .safetensors file into any of those locations."
        raise FileNotFoundError(error_message)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


# ============================================================================
# State Dict Transformations
# ============================================================================

checkpoint_dict_replacements_sd1 = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_replacements_sd2_turbo = {
    'conditioner.embedders.0.': 'cond_stage_model.',
}


def transform_checkpoint_dict_key(k: str, replacements: dict) -> str:
    for text, replacement in replacements.items():
        if k.startswith(text):
            return replacement + k[len(text):]
    return k


def get_state_dict_from_checkpoint(pl_sd: dict) -> dict:
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    is_sd2_turbo = (
        'conditioner.embedders.0.model.ln_final.weight' in pl_sd and
        pl_sd['conditioner.embedders.0.model.ln_final.weight'].size()[0] == 1024
    )

    replacements = checkpoint_dict_replacements_sd2_turbo if is_sd2_turbo else checkpoint_dict_replacements_sd1

    sd = {
        transform_checkpoint_dict_key(k, replacements): v
        for k, v in pl_sd.items()
    }

    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


def read_metadata_from_safetensors(filename: str) -> dict:
    """Read metadata from safetensors file header."""
    import json

    with open(filename, mode="rb") as file:
        metadata_len = int.from_bytes(file.read(8), "little")
        json_start = file.read(2)

        if metadata_len <= 2 or json_start not in (b'{"', b"{'"):
            raise ValueError(f"{filename} is not a valid safetensors file")

        res = {}
        try:
            json_data = json_start + file.read(metadata_len - 2)
            json_obj = json.loads(json_data)

            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v.startswith('{'):
                    try:
                        res[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            errors.report(f"Error reading metadata from {filename}", exc_info=True)

        return res


def read_state_dict(checkpoint_file: str, print_global_state: bool = False,
                    map_location: Optional[str] = None) -> dict:
    _, extension = os.path.splitext(checkpoint_file)

    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()

        if not shared.opts.disable_mmap_load_safetensors:
            pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
        else:
            with open(checkpoint_file, 'rb') as f:
                pl_sd = safetensors.torch.load(f.read())
            pl_sd = {k: v.to(device) for k, v in pl_sd.items()}
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    return get_state_dict_from_checkpoint(pl_sd)


# ============================================================================
# Memory Management
# ============================================================================

def complete_model_teardown(model) -> None:
    """Tear down a model by breaking references to its components."""
    if model is None:
        return

    model_name = _get_model_name(model)
    print(f"Performing complete teardown of model: {model_name}")

    # Build set of objects to preserve
    preserve_ids = _get_preserve_ids(model)

    # Process the model's attributes
    visited = set()
    for attr_name in dir(model):
        if attr_name.startswith('__'):
            continue
        try:
            attr = getattr(model, attr_name, None)
            if attr is not None:
                if isinstance(attr, (dict, list, tuple)) or hasattr(attr, 'parameters'):
                    _teardown_recursive(attr, preserve_ids, visited)
                    if id(attr) not in preserve_ids:
                        try:
                            setattr(model, attr_name, None)
                        except (AttributeError, TypeError):
                            pass
        except Exception:
            pass

    # Cleanup
    _force_cuda_cleanup()
    _run_gc(3)
    print("Model teardown completed")


def _get_model_name(model) -> str:
    """Get model name for logging."""
    if hasattr(model, 'sd_checkpoint_info') and hasattr(model.sd_checkpoint_info, 'title'):
        return model.sd_checkpoint_info.title
    if hasattr(model, 'filename'):
        return model.filename
    return "Unknown"


def _get_preserve_ids(model) -> Set[int]:
    """Get IDs of objects that should be preserved during teardown."""
    preserve_ids = set()

    if not hasattr(model, 'forge_objects'):
        return preserve_ids

    preserve_ids.add(id(model.forge_objects))

    if hasattr(model.forge_objects, 'vae'):
        vae = model.forge_objects.vae
        preserve_ids.add(id(vae))

        if hasattr(vae, 'model'):
            preserve_ids.add(id(vae.model))
            for attr in ('encoder', 'decoder', 'quantize'):
                if hasattr(vae.model, attr):
                    preserve_ids.add(id(getattr(vae.model, attr)))

        if hasattr(vae, 'patcher'):
            preserve_ids.add(id(vae.patcher))

    return preserve_ids


def _teardown_recursive(obj, preserve_ids: Set[int], visited: Set[int], depth: int = 0) -> None:
    """Recursively teardown object tree."""
    if depth > 10 or obj is None:
        return

    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    try:
        if hasattr(obj, 'named_parameters'):
            _teardown_module(obj, preserve_ids, visited, depth)
        elif isinstance(obj, dict):
            _teardown_dict(obj, preserve_ids, visited, depth)
        elif isinstance(obj, (list, tuple)):
            _teardown_sequence(obj, preserve_ids, visited, depth)
    except Exception:
        pass


def _teardown_module(obj, preserve_ids: Set[int], visited: Set[int], depth: int) -> None:
    """Teardown torch.nn.Module."""
    obj_id = id(obj)

    # Handle parameters
    try:
        for name, param in list(obj.named_parameters(recurse=False)):
            if hasattr(param, 'data') and obj_id not in preserve_ids:
                try:
                    param.data = torch.empty(0)
                except Exception:
                    pass
    except Exception:
        pass

    # Handle buffers
    try:
        for name, buffer in list(obj.named_buffers(recurse=False)):
            if hasattr(buffer, 'data') and obj_id not in preserve_ids:
                try:
                    buffer.data = torch.empty(0)
                except Exception:
                    pass
    except Exception:
        pass

    # Process children
    try:
        for name, module in list(obj.named_children()):
            _teardown_recursive(module, preserve_ids, visited, depth + 1)
            if id(obj) not in preserve_ids and id(module) not in preserve_ids:
                try:
                    setattr(obj, name, None)
                except Exception:
                    pass
    except Exception:
        pass


def _teardown_dict(obj: dict, preserve_ids: Set[int], visited: Set[int], depth: int) -> None:
    """Teardown dictionary."""
    obj_id = id(obj)
    for key in list(obj.keys()):
        try:
            val = obj[key]
            if hasattr(val, 'parameters') or hasattr(val, 'numel'):
                _teardown_recursive(val, preserve_ids, visited, depth + 1)
                if obj_id not in preserve_ids and id(val) not in preserve_ids:
                    obj[key] = None
        except Exception:
            pass

    if obj_id not in preserve_ids:
        try:
            obj.clear()
        except Exception:
            pass


def _teardown_sequence(obj, preserve_ids: Set[int], visited: Set[int], depth: int) -> None:
    """Teardown list/tuple."""
    for item in obj:
        try:
            if hasattr(item, 'parameters') or hasattr(item, 'numel'):
                _teardown_recursive(item, preserve_ids, visited, depth + 1)
        except Exception:
            pass


def _force_cuda_cleanup() -> None:
    """Force CUDA cache cleanup."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _run_gc(passes: int = 3) -> None:
    """Run garbage collection multiple times."""
    for _ in range(passes):
        gc.collect()


def force_memory_deallocation() -> str:
    """Force deallocation of memory."""
    global _models_loaded_count, _peak_memory_usage

    process = psutil.Process()
    pre_mem = process.memory_info().rss

    # Clear caches
    if checkpoints_loaded:
        print(f"Clearing {len(checkpoints_loaded)} cached state dictionaries")
        checkpoints_loaded.clear()

    # Minimize metadata in checkpoints_list
    for info in checkpoints_list.values():
        if hasattr(info, 'metadata') and info.metadata:
            minimal = {}
            if 'ss_sd_model_name' in info.metadata:
                minimal['ss_sd_model_name'] = info.metadata['ss_sd_model_name']
            info.metadata = minimal

    _force_cuda_cleanup()

    # Track large tensors - fixed safe iteration
    tracked_refs = []
    try:
        objects = list(gc.get_objects())  # Create a copy to iterate safely
        for obj in objects:
            try:
                if isinstance(obj, torch.Tensor) and obj.numel() > 1e6:
                    if hasattr(obj, 'is_cuda') and not obj.is_cuda:
                        tracked_refs.append(weakref.ref(obj))
            except (ReferenceError, RuntimeError):
                continue
        del objects
    except Exception:
        pass

    # Run GC
    for i in range(3):
        count = gc.collect()
        if count == 0:
            break
        print(f"GC pass {i+1}: collected {count} objects")

    post_mem = process.memory_info().rss
    mem_diff = (post_mem - pre_mem) / (1024 * 1024)

    _models_loaded_count += 1
    _peak_memory_usage = max(_peak_memory_usage, post_mem)

    print(f"Memory change: {mem_diff:.2f} MB ({post_mem/(1024**3):.2f} GB total)")
    return f"Memory cleanup: {post_mem/(1024**3):.2f} GB used"


def aggressive_memory_cleanup() -> str:
    """Perform aggressive memory cleanup."""
    print("Performing aggressive memory cleanup...")

    if checkpoints_loaded:
        print(f"Clearing {len(checkpoints_loaded)} cached checkpoints")
        checkpoints_loaded.clear()

    _force_cuda_cleanup()

    collected = gc.collect()
    print(f"GC: collected {collected} objects")

    mem_info = psutil.Process().memory_info()
    print(f"Current memory: RSS={mem_info.rss/(1024**3):.2f} GB, VMS={mem_info.vms/(1024**3):.2f} GB")

    return f"Cleanup complete. Usage: {mem_info.rss/(1024**3):.2f} GB"


# ============================================================================
# Checkpoint Loading
# ============================================================================

def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer: Timer) -> dict:
    """Load state dict from checkpoint file."""
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")

    device = shared.weight_load_location or model_management.get_torch_device()

    if checkpoint_info.is_safetensors:
        if shared.opts.disable_mmap_load_safetensors:
            with torch.no_grad():
                with open(checkpoint_info.filename, 'rb') as f:
                    data = f.read()
                res = safetensors.torch.load(data)
                res = {k: v.to(device) for k, v in res.items()}
                del data
        else:
            res = safetensors.torch.load_file(checkpoint_info.filename, device=device)
    else:
        res = torch.load(checkpoint_info.filename, map_location=device)
        res = get_state_dict_from_checkpoint(res)

    timer.record("load weights from disk")
    return res


class SkipWritingToConfig:
    """Context manager to prevent writing checkpoint name to config."""
    skip = False
    previous = None

    def __enter__(self):
        self.previous = SkipWritingToConfig.skip
        SkipWritingToConfig.skip = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        SkipWritingToConfig.skip = self.previous


# ============================================================================
# Model Configuration
# ============================================================================

def check_fp8(model) -> Optional[bool]:
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        return False
    if shared.opts.fp8_storage == "Enable":
        return True
    if getattr(model, "is_sdxl", False) and shared.opts.fp8_storage == "Enable for SDXL":
        return True
    return False


def set_model_type(model, state_dict: dict) -> None:
    """Set model type flags based on state dict keys."""
    model.is_sd1 = False
    model.is_sd2 = False
    model.is_sdxl = False
    model.is_ssd = False
    model.is_sd3 = False

    if "model.diffusion_model.x_embedder.proj.weight" in state_dict:
        model.is_sd3 = True
        model.model_type = ModelType.SD3
    elif hasattr(model, 'conditioner'):
        model.is_sdxl = True
        if 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in state_dict:
            model.is_ssd = True
            model.model_type = ModelType.SSD
        else:
            model.model_type = ModelType.SDXL
    elif hasattr(model.cond_stage_model, 'model'):
        model.is_sd2 = True
        model.model_type = ModelType.SD2
    else:
        model.is_sd1 = True
        model.model_type = ModelType.SD1


def set_model_fields(model) -> None:
    """Set latent channel/dimension fields on model."""
    if hasattr(model, "latent_format"):
        model.latent_channels = getattr(model.latent_format, "latent_channels", 4)
        model.latent_dimensions = getattr(model.latent_format, "latent_dimensions", 2)
    elif hasattr(model, "model") and hasattr(model.model, "latent_format"):
        lf = model.model.latent_format
        model.latent_channels = getattr(lf, "latent_channels", 4)
        model.latent_dimensions = getattr(lf, "latent_dimensions", 2)
    elif not hasattr(model, 'latent_channels'):
        model.latent_channels = 4


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict: dict, timer: Timer) -> None:
    """Stub - weights loaded via forge_loader."""
    pass


# ============================================================================
# Midas Setup
# ============================================================================

def enable_midas_autodownload() -> None:
    """Enable automatic downloading of midas models."""
    midas_path = os.path.join(paths.models_path, 'midas')

    for k, v in midas.api.ISL_PATHS.items():
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, os.path.basename(v))

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            os.makedirs(midas_path, exist_ok=True)
            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")
        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def patch_given_betas() -> None:
    """Patch register_schedule to handle OmegaConf lists."""
    import ldm.models.diffusion.ddpm

    def patched_register_schedule(*args, **kwargs):
        if isinstance(args[1], ListConfig):
            args = (args[0], np.array(args[1]), *args[2:])
        original_register_schedule(*args, **kwargs)

    original_register_schedule = patches.patch(
        __name__, ldm.models.diffusion.ddpm.DDPM,
        'register_schedule', patched_register_schedule
    )


def repair_config(sd_config, state_dict: Optional[dict] = None) -> None:
    """Repair/adjust model config for compatibility."""
    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    if hasattr(sd_config.model.params, 'unet_config'):
        unet_params = sd_config.model.params.unet_config.params
        if shared.cmd_opts.no_half:
            unet_params.use_fp16 = False
        elif shared.cmd_opts.upcast_sampling or shared.cmd_opts.precision == "half":
            unet_params.use_fp16 = True

    if hasattr(sd_config.model.params, 'first_stage_config'):
        ddconfig = sd_config.model.params.first_stage_config.params.ddconfig
        if getattr(ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
            ddconfig.attn_type = "vanilla"

    # Override karlo path for UnCLIP-L
    if hasattr(sd_config.model.params, "noise_aug_config"):
        noise_config = sd_config.model.params.noise_aug_config
        if hasattr(noise_config.params, "clip_stats_path"):
            karlo_path = os.path.join(paths.models_path, 'karlo')
            noise_config.params.clip_stats_path = noise_config.params.clip_stats_path.replace(
                "checkpoints/karlo_models", karlo_path
            )

    # Disable checkpointing for inference
    for config_path in ["network_config", "unet_config"]:
        if hasattr(sd_config.model.params, config_path):
            getattr(sd_config.model.params, config_path).params.use_checkpoint = False


# ============================================================================
# Alpha Schedule
# ============================================================================

def rescale_zero_terminal_snr_abar(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt ** 2
    alphas_bar[-1] = 4.8973451890853435e-08
    return alphas_bar


def apply_alpha_schedule_override(sd_model, p=None, force_apply: bool = False) -> None:
    """Apply alpha schedule overrides based on settings."""
    if not hasattr(sd_model, 'alphas_cumprod') or not hasattr(sd_model, 'alphas_cumprod_original'):
        return

    sd_model.alphas_cumprod = sd_model.alphas_cumprod_original.to(shared.device)

    if shared.opts.use_downcasted_alpha_bar:
        if p is not None:
            p.extra_generation_params['Downcast alphas_cumprod'] = shared.opts.use_downcasted_alpha_bar
        sd_model.alphas_cumprod = sd_model.alphas_cumprod.half().to(shared.device)

    should_apply_ztsnr = (
        shared.opts.sd_noise_schedule == "Zero Terminal SNR" or
        getattr(sd_model, 'ztsnr', False) or
        force_apply
    )

    if should_apply_ztsnr:
        if p is not None and shared.opts.sd_noise_schedule != "Default":
            p.extra_generation_params['Noise Schedule'] = shared.opts.sd_noise_schedule
        sd_model.alphas_cumprod = rescale_zero_terminal_snr_abar(sd_model.alphas_cumprod).to(shared.device)


# ============================================================================
# Model Data Management
# ============================================================================

sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight = 'conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight = 'conditioner.embedders.0.model.ln_final.weight'


class SdModelData:
    def __init__(self):
        self.sd_model = None
        self.loaded_sd_models: list = []
        self.was_loaded_at_least_once = False
        self.lock = threading.Lock()

    def get_sd_model(self):
        if self.was_loaded_at_least_once:
            return self.sd_model

        if self.sd_model is None:
            with self.lock:
                if self.sd_model is not None or self.was_loaded_at_least_once:
                    return self.sd_model
                try:
                    load_model()
                except Exception as e:
                    errors.display(e, "loading stable diffusion model", full_traceback=True)
                    print("Stable diffusion model failed to load", file=sys.stderr)
                    self.sd_model = None

        return self.sd_model

    def set_sd_model(self, v, already_loaded: bool = False) -> None:
        self.sd_model = v
        if already_loaded and v is not None:
            sd_vae.base_vae = getattr(v, "base_vae", None)
            sd_vae.loaded_vae_file = getattr(v, "loaded_vae_file", None)
            sd_vae.checkpoint_info = v.sd_checkpoint_info


model_data = SdModelData()


def get_empty_cond(sd_model):
    p = processing.StableDiffusionProcessingTxt2Img()
    extra_networks.activate(p, {})

    if hasattr(sd_model, 'get_learned_conditioning'):
        d = sd_model.get_learned_conditioning([""])
    else:
        d = sd_model.cond_stage_model([""])

    if isinstance(d, dict):
        d = d['crossattn']

    return d


def send_model_to_cpu(m) -> None:
    """Stub - handled by model_management."""
    pass


def model_target_device(m):
    return devices.device


def send_model_to_device(m) -> None:
    """Stub - handled by model_management."""
    pass


def send_model_to_trash(m) -> None:
    """Stub - handled by complete_model_teardown."""
    pass


def instantiate_from_config(config, state_dict: Optional[dict] = None):
    constructor = get_obj_from_str(config["target"])
    params = {**config.get("params", {})}

    if state_dict and "state_dict" in params and params["state_dict"] is None:
        params["state_dict"] = state_dict

    return constructor(**params)


def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# ============================================================================
# Main Model Loading Functions
# ============================================================================

def load_model(checkpoint_info: Optional[CheckpointInfo] = None,
               already_loaded_state_dict: Optional[dict] = None,
               forced_reload: bool = False):
    """Load a model from checkpoint."""
    global model_data

    checkpoint_info = checkpoint_info or select_checkpoint()
    timer = Timer()

    # Check if already loaded
    if not forced_reload:
        for loaded_model in model_data.loaded_sd_models:
            if loaded_model.filename == checkpoint_info.filename:
                log.debug(f"Using already loaded model {loaded_model.sd_checkpoint_info.title}")
                model_data.loaded_sd_models.remove(loaded_model)
                model_data.loaded_sd_models.insert(0, loaded_model)
                model_data.set_sd_model(loaded_model, already_loaded=True)
                return loaded_model
    else:
        # Remove existing model with same filename
        for loaded_model in list(model_data.loaded_sd_models):
            if loaded_model.filename == checkpoint_info.filename:
                print(f"Forced reload: Unloading existing model {loaded_model.sd_checkpoint_info.title}")
                model_data.loaded_sd_models.remove(loaded_model)
                complete_model_teardown(loaded_model)
                break

    # Enforce model limit
    while len(model_data.loaded_sd_models) >= shared.opts.sd_checkpoints_limit:
        unload_first_loaded_model()

    force_memory_deallocation()
    timer.record("memory cleanup")

    print(f"Loading model {checkpoint_info.title} ({len(model_data.loaded_sd_models) + 1} of {shared.opts.sd_checkpoints_limit})")

    sd_model = None
    state_dict = None

    try:
        state_dict = already_loaded_state_dict or get_checkpoint_state_dict(checkpoint_info, timer)
        sd_model = forge_loader.load_model_for_a1111(timer=timer, checkpoint_info=checkpoint_info, state_dict=state_dict)
        sd_model.filename = checkpoint_info.filename
    finally:
        if state_dict is not None:
            del state_dict
            gc.collect()

    if sd_model is None:
        print("Error: Model failed to load")
        return None

    # Setup model
    model_data.loaded_sd_models.insert(0, sd_model)
    model_data.set_sd_model(sd_model)
    model_data.was_loaded_at_least_once = True

    # VAE protection
    vae_preserver = VAEStructurePreserver()
    sd_model.forge_objects.vae_preserver = vae_preserver

    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    # Load VAE
    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
    sd_vae.load_vae(sd_model, vae_file, vae_source)
    timer.record("load VAE")

    timer.record("load textual inversion embeddings")

    script_callbacks.model_loaded_callback(sd_model)
    timer.record("scripts callbacks")

    # Apply text encoder
    try:
        from modules import sd_text_encoder
        if not getattr(sd_model, '_separate_te_loaded', False):
            sd_text_encoder.apply_text_encoder()
        else:
            print("Skipping apply_text_encoder() - separate TE already loaded")
        timer.record("apply text encoder")
    except Exception as e:
        log.warning(f"Error applying text encoder: {e}")

    with torch.no_grad():
        sd_model.cond_stage_model_empty_prompt = get_empty_cond(sd_model)
    timer.record("calculate empty prompt")

    print(f"Model {checkpoint_info.title} loaded in {timer.summary()}.")

    force_memory_deallocation()

    return sd_model


def set_model_active(model_index) -> str:
    """Set a specific model as active by index."""
    try:
        model_index = int(model_index)
    except (TypeError, ValueError):
        return "Model index must be a number"

    if not model_data.loaded_sd_models:
        return "No models currently loaded"

    if not 0 <= model_index < len(model_data.loaded_sd_models):
        return f"Invalid model index: {model_index}, valid range is 0-{len(model_data.loaded_sd_models)-1}"

    model_to_activate = model_data.loaded_sd_models[model_index]

    if model_data.sd_model == model_to_activate:
        return f"Model {model_to_activate.sd_checkpoint_info.title} is already active"

    model_data.loaded_sd_models.remove(model_to_activate)
    model_data.loaded_sd_models.insert(0, model_to_activate)
    model_data.set_sd_model(model_to_activate, already_loaded=True)

    return f"Activated model: {model_to_activate.sd_checkpoint_info.title}"


def unload_first_loaded_model() -> None:
    """Unload the oldest loaded model."""
    if not model_data.loaded_sd_models:
        return

    first_loaded_model = model_data.loaded_sd_models.pop(-1)
    model_name = _get_model_name(first_loaded_model)
    print(f"Unloading first loaded model: {model_name}")

    complete_model_teardown(first_loaded_model)
    del first_loaded_model

    _run_gc(2)

    mem_info = psutil.Process().memory_info()
    print(f"Memory after unload: RSS={mem_info.rss/(1024**3):.2f} GB")


def reuse_model_from_already_loaded(sd_model, checkpoint_info: CheckpointInfo, timer: Timer) -> None:
    """Stub - handled in load_model."""
    pass


def reload_model_weights(sd_model=None, info: Optional[CheckpointInfo] = None,
                         forced_reload: bool = False):
    return load_model(info, forced_reload=forced_reload)


def unload_model_weights(model=None) -> str:
    """Unload the currently active model to RAM."""
    model = model or model_data.sd_model

    if model is None:
        return "No model is currently loaded"

    print(f"Unloading model weights for {model.sd_checkpoint_info.title}")

    if hasattr(model, 'model_unload'):
        model.model_unload()
    elif hasattr(model, 'offload_device'):
        model.to(model.offload_device)
    else:
        model.to('cpu')

    model_management.soft_empty_cache(force=True)
    gc.collect()

    return f"Unloaded model {model.sd_checkpoint_info.title} to RAM"


def load_model_to_device(model=None) -> str:
    """Load a model from RAM to VRAM."""
    model = model or model_data.sd_model

    if model is None:
        return "No model is currently loaded"

    print(f"Loading model weights for {model.sd_checkpoint_info.title} to device")

    if hasattr(model, 'model_load'):
        model.model_load()
    else:
        model.to(model_management.get_torch_device())

    return f"Loaded model {model.sd_checkpoint_info.title} to device"


def list_loaded_models() -> str:
    """Return a list of all currently loaded models."""
    if not model_data.loaded_sd_models:
        return "No models currently loaded"

    total_ram = psutil.Process().memory_info().rss / (1024**3)

    lines = [f"Currently loaded models (Total RAM: {total_ram:.2f} GB):"]
    for i, model in enumerate(model_data.loaded_sd_models):
        active = " (active)" if model == model_data.sd_model else ""
        lines.append(f"[{i}] {model.sd_checkpoint_info.title}{active}")

    return "\n".join(lines)


def unload_specific_model(model_index) -> str:
    """Unload a specific model by index."""
    try:
        model_index = int(model_index)
    except (TypeError, ValueError):
        return "Model index must be a number"

    if not model_data.loaded_sd_models:
        return "No models currently loaded"

    if not 0 <= model_index < len(model_data.loaded_sd_models):
        return f"Invalid model index: {model_index}, valid range is 0-{len(model_data.loaded_sd_models)-1}"

    model_to_unload = model_data.loaded_sd_models[model_index]
    name = model_to_unload.sd_checkpoint_info.title
    is_active = model_to_unload == model_data.sd_model

    # Switch active model if needed
    if is_active and len(model_data.loaded_sd_models) > 1:
        new_index = 0 if model_index != 0 else 1
        new_active = model_data.loaded_sd_models[new_index]
        print(f"Switching active model from {name} to {new_active.sd_checkpoint_info.title}")
        model_data.set_sd_model(new_active, already_loaded=True)

    model_data.loaded_sd_models.pop(model_index)

    # Unload
    if hasattr(model_to_unload, 'model_unload'):
        model_to_unload.model_unload()
    elif hasattr(model_to_unload, 'offload_device'):
        model_to_unload.to(model_to_unload.offload_device)
    else:
        model_to_unload.to('cpu')

    model_management.soft_empty_cache(force=True)
    gc.collect()

    status = f"Unloaded model: {name}"
    if is_active and not model_data.loaded_sd_models:
        status += "\nWarning: No active model remaining"

    return status


def apply_token_merging(sd_model, token_merging_ratio: float) -> None:
    """Apply token merging optimization to model."""
    if token_merging_ratio <= 0:
        return

    print(f'token_merging_ratio = {token_merging_ratio}')

    from ldm_patched.contrib.nodes_tomesd import TomePatcher
    sd_model.forge_objects.unet = TomePatcher().patch(
        model=sd_model.forge_objects.unet,
        ratio=token_merging_ratio
    )
