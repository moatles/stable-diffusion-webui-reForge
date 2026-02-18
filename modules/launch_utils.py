#!/usr/bin/env python3
"""
Installation script for stable-diffusion-webui-reForge.
Installs necessary requirements and launches main program in webui.py
"""
import logging
import re
import subprocess
import os
import shutil
import sys
import importlib.util
import importlib.metadata
import json
import shlex
from functools import lru_cache
from typing import NamedTuple, Optional
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from modules import cmd_args, errors
from modules.paths_internal import script_path, extensions_dir, extensions_builtin_dir
from modules.timer import startup_timer
from modules import logging_config
from modules_forge import forge_version
from modules_forge.config import always_disabled_extensions

args, _ = cmd_args.parser.parse_known_args()
logging_config.setup_logging(args.loglevel)

python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
dir_repos = "repositories"

default_command_live = os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1"

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')


# GPU Detection Cache
_gpu_info_cache: Optional[dict] = None


def detect_gpu() -> dict:
    """
    Detect GPU vendor and model.
    Returns dict with keys: vendor ('amd', 'nvidia', 'intel', 'none'), model, vram_mb
    """
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return _gpu_info_cache

    gpu_info = {
        'vendor': 'none',
        'model': 'Unknown',
        'vram_mb': 0,
        'is_amd': False,
        'is_nvidia': False,
    }

    # Try AMD detection first (via rocm-smi or lspci)
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                if 'card series' in line.lower() or 'gpu' in line.lower():
                    gpu_info['vendor'] = 'amd'
                    gpu_info['is_amd'] = True
                    gpu_info['model'] = line.split(':')[-1].strip() if ':' in line else line.strip()
                    break
            if gpu_info['is_amd']:
                vram_result = subprocess.run(
                    ['rocm-smi', '--showmeminfo', 'vram'],
                    capture_output=True, text=True, timeout=5
                )
                if vram_result.returncode == 0:
                    for line in vram_result.stdout.splitlines():
                        if 'total' in line.lower():
                            match = re.search(r'(\d+)', line)
                            if match:
                                gpu_info['vram_mb'] = int(match.group(1)) // (1024 * 1024)
                                break
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # Try AMD detection via lspci if rocm-smi failed
    if not gpu_info['is_amd']:
        try:
            result = subprocess.run(
                ['lspci'], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line_lower = line.lower()
                    if ('vga' in line_lower or '3d' in line_lower) and ('amd' in line_lower or 'radeon' in line_lower):
                        gpu_info['vendor'] = 'amd'
                        gpu_info['is_amd'] = True
                        gpu_info['model'] = line.split(':')[-1].strip() if ':' in line else line.strip()
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

    # Try NVIDIA detection if not AMD
    if not gpu_info['is_amd']:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout:
                line = result.stdout.strip().split('\n')[0]
                parts = line.split(',')
                gpu_info['vendor'] = 'nvidia'
                gpu_info['is_nvidia'] = True
                gpu_info['model'] = parts[0].strip()
                if len(parts) > 1:
                    vram_str = parts[1].strip()
                    match = re.search(r'(\d+)', vram_str)
                    if match:
                        gpu_info['vram_mb'] = int(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

    _gpu_info_cache = gpu_info
    return gpu_info


def is_amd_gpu() -> bool:
    """Check if system has AMD GPU."""
    return detect_gpu()['is_amd']


def is_nvidia_gpu() -> bool:
    """Check if system has NVIDIA GPU."""
    return detect_gpu()['is_nvidia']


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if not (major == 3 and 7 <= minor <= 13):
        errors.print_error_explanation(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.19 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI's directory.

On arch/cachyos, these will setup the environment:
  source venv/bin/activate.fish
  yay -S python310
  uv pip install "setuptools<70.0.0"
  uv pip install --no-build-isolation https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip

Use --skip-python-version-check to suppress this warning.
""")


@lru_cache()
def commit_hash():
    try:
        return subprocess.check_output(
            [git, "-C", script_path, "rev-parse", "HEAD"],
            encoding='utf8', stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "<none>"


@lru_cache()
def git_tag_a1111():
    try:
        return subprocess.check_output(
            [git, "-C", script_path, "describe", "--tags"],
            encoding='utf8', stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        try:
            changelog_md = os.path.join(script_path, "CHANGELOG.md")
            with open(changelog_md, "r", encoding="utf-8") as file:
                line = next((l.strip() for l in file if l.strip()), "<none>")
                return line.replace("## ", "")
        except Exception:
            return "<none>"


def git_tag():
    return f'f{forge_version.version}-{git_tag_a1111()}'


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def is_installed(package):
    try:
        return importlib.metadata.distribution(package) is not None
    except importlib.metadata.PackageNotFoundError:
        try:
            return importlib.util.find_spec(package) is not None
        except ModuleNotFoundError:
            return False


def get_package_version(package):
    """Get installed version of a package, or None if not installed."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def compare_versions(v1, v2):
    """
    Compare two version strings without requiring packaging module.
    Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    def normalize(v):
        parts = []
        for part in re.split(r'[.\-_]', v):
            numeric = re.match(r'^(\d+)', part)
            if numeric:
                parts.append(int(numeric.group(1)))
            else:
                parts.append(0)
        return parts

    v1_parts = normalize(v1)
    v2_parts = normalize(v2)

    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))

    for a, b in zip(v1_parts, v2_parts):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live):
    """Run a uv pip command (named run_pip for backward compatibility)."""
    if args.skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url else ''
    return run(
        f'uv pip {command}{index_url_line}',
        desc=f"Installing {desc}",
        errdesc=f"Couldn't install {desc}",
        live=live
    )


# Alias for explicit uv usage
run_uv = run_pip


def check_run_python(code: str) -> bool:
    result = subprocess.run([python, "-c", code], capture_output=True)
    return result.returncode == 0


def git_fix_workspace(directory, name):
    run(f'{git} -C "{directory}" fetch --refetch --no-auto-gc',
        f"Fetching all contents for {name}",
        f"Couldn't fetch {name}", live=True)
    run(f'{git} -C "{directory}" gc --aggressive --prune=now',
        f"Pruning {name}",
        f"Couldn't prune {name}", live=True)


def run_git(directory, name, command, desc=None, errdesc=None, custom_env=None,
            live: bool = default_command_live, autofix=True):
    try:
        return run(f'{git} -C "{directory}" {command}', desc=desc, errdesc=errdesc,
                   custom_env=custom_env, live=live)
    except RuntimeError:
        if not autofix:
            raise

    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(directory, name)
    return run(f'{git} -C "{directory}" {command}', desc=desc, errdesc=errdesc,
               custom_env=custom_env, live=live)


def git_clone(url, directory, name, commithash=None):
    if os.path.exists(directory):
        if commithash is None:
            return

        current_hash = run_git(directory, name, 'rev-parse HEAD', None,
                               f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        if current_hash == commithash:
            return

        origin_url = run_git(directory, name, 'config --get remote.origin.url', None,
                             f"Couldn't determine {name}'s origin URL", live=False).strip()
        if origin_url != url:
            run_git(directory, name, f'remote set-url origin "{url}"', None,
                    f"Failed to set {name}'s origin URL", live=False)

        run_git(directory, name, 'fetch', f"Fetching updates for {name}...",
                f"Couldn't fetch {name}", autofix=False)
        run_git(directory, name, f'checkout {commithash}',
                f"Checking out commit for {name} with hash: {commithash}...",
                f"Couldn't checkout commit {commithash} for {name}", live=True)
        return

    try:
        run(f'{git} clone --config core.filemode=false "{url}" "{directory}"',
            f"Cloning {name} into {directory}...",
            f"Couldn't clone {name}", live=True)
    except RuntimeError:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    if commithash is not None:
        run(f'{git} -C "{directory}" checkout {commithash}', None,
            f"Couldn't checkout {name}'s hash: {commithash}")


def git_pull_recursive(directory):
    for subdir, _, _ in os.walk(directory):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def version_check(commit):
    try:
        import requests
        commits = requests.get(
            'https://api.github.com/repos/moatles/stable-diffusion-webui-reForge/branches/main',
            timeout=10
        ).json()
        if commit != "<none>" and commits['commit']['sha'] != commit:
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits['commit']['sha'] == commit:
            print("You are up to date with the most recent release.")
        else:
            print("Not a git clone, can't perform version check.")
    except Exception as e:
        print(f"version check failed: {e}")


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        stdout = run(f'"{python}" "{path_installer}"',
                     errdesc=f"Error running install.py for extension {extension_dir}",
                     custom_env=env).strip()
        if stdout:
            print(stdout)
    except Exception as e:
        errors.report(str(e))


def load_settings(settings_file):
    """Load settings from a JSON file, handling errors gracefully."""
    try:
        with open(settings_file, "r", encoding="utf8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except Exception:
        errors.report(
            f'\nCould not load settings\n'
            f'The config file "{settings_file}" is likely corrupted\n'
            f'It has been moved to "tmp/config.json"\n'
            f'Reverting config to default\n\n',
            exc_info=True
        )
        tmp_dir = os.path.join(script_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        os.replace(settings_file, os.path.join(tmp_dir, "config.json"))
        return {}


def list_extensions(settings_file):
    """List enabled extensions from the extensions directory."""
    settings = load_settings(settings_file)

    disabled_extensions = set(settings.get('disabled_extensions', []) + always_disabled_extensions)
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if (disable_all_extensions != 'none' or
            args.disable_extra_extensions or
            args.disable_all_extensions or
            not os.path.isdir(extensions_dir)):
        return []

    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions]


def list_extensions_builtin(settings_file):
    """List enabled extensions from the builtin extensions directory."""
    settings = load_settings(settings_file)

    disabled_extensions = set(settings.get('disabled_extensions', []))
    disable_all_extensions = settings.get('disable_all_extensions', 'none')

    if (disable_all_extensions != 'none' or
            args.disable_extra_extensions or
            args.disable_all_extensions or
            not os.path.isdir(extensions_builtin_dir)):
        return []

    return [x for x in os.listdir(extensions_builtin_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if os.path.isdir(extensions_dir):
        with startup_timer.subcategory("run extensions installers"):
            for dirname_extension in list_extensions(settings_file):
                logging.debug(f"Installing {dirname_extension}")
                path = os.path.join(extensions_dir, dirname_extension)
                if os.path.isdir(path):
                    run_extension_installer(path)
                    startup_timer.record(dirname_extension)

    if os.path.isdir(extensions_builtin_dir):
        with startup_timer.subcategory("run extensions_builtin installers"):
            for dirname_extension in list_extensions_builtin(settings_file):
                logging.debug(f"Installing {dirname_extension}")
                path = os.path.join(extensions_builtin_dir, dirname_extension)
                if os.path.isdir(path):
                    run_extension_installer(path)
                    startup_timer.record(dirname_extension)


RE_REQUIREMENT = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def requirements_met(requirements_file):
    """
    Parse a requirements.txt file to determine if all requirements are installed.
    Returns True if so, False if not installed or parsing fails.
    """
    try:
        with open(requirements_file, "r", encoding="utf8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                m = RE_REQUIREMENT.match(line)
                if m is None:
                    return False

                package = m.group(1).strip()
                version_required = (m.group(2) or "").strip()

                if not version_required:
                    continue

                version_installed = get_package_version(package)
                if version_installed is None:
                    return False

                if compare_versions(version_required, version_installed) != 0:
                    return False
    except FileNotFoundError:
        return False

    return True


def get_cuda_compute_cap():
    """
    Returns float of CUDA Compute Capability using nvidia-smi.
    Returns 0.0 on error or if not NVIDIA GPU.
    """
    if is_amd_gpu():
        return 0.0

    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=noheader,csv'],
            text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        return max(map(float, filter(None, output)))
    except Exception:
        return 0.0


def install_setuptools():
    """Install setuptools < 70.0.0 if needed."""
    setuptools_version = get_package_version("setuptools")
    needs_setuptools = False

    if setuptools_version is None:
        needs_setuptools = True
        print("setuptools not found, will install")
    else:
        if compare_versions(setuptools_version, "70.0.0") >= 0:
            needs_setuptools = True
            print(f"setuptools {setuptools_version} >= 70.0.0, will downgrade")

    if needs_setuptools:
        run(
            'uv pip install "setuptools<70.0.0"',
            desc="Installing setuptools<70.0.0 (required for package builds)",
            errdesc="Couldn't install setuptools",
            live=True
        )
        startup_timer.record("install setuptools")


def install_clip():
    """Install CLIP with --no-build-isolation (must be called after torch is installed)."""
    if is_installed("clip"):
        return

    clip_package = os.environ.get(
        'CLIP_PACKAGE',
        "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
    )

    run(
        f'uv pip install --no-build-isolation --no-deps {clip_package}',
        desc="Installing CLIP (required dependency)",
        errdesc="Couldn't install CLIP",
        live=True
    )
    startup_timer.record("install clip")


def prepare_environment():
    gpu_info = detect_gpu()
    print(f"Detected GPU: {gpu_info['vendor'].upper()} - {gpu_info['model']}")
    if gpu_info['vram_mb'] > 0:
        print(f"VRAM: {gpu_info['vram_mb']} MB")

    # Set torch command based on GPU vendor
    if is_amd_gpu():
        # AMD GPU - use ROCm
        torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/nightly/rocm6.3")
        torch_command = os.environ.get(
            'TORCH_COMMAND',
            f"uv pip install torch torchvision --index-url {torch_index_url}"
        )
        xformers_package = None
    else:
        # NVIDIA GPU or fallback - use CUDA
        torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu128")
        torch_command = os.environ.get(
            'TORCH_COMMAND',
            f"uv pip install torch==2.9.0 torchvision --index-url {torch_index_url}"
        )
        xformers_package = os.environ.get(
            'XFORMERS_PACKAGE',
            'xformers --index-url https://download.pytorch.org/whl/cu128'
        )

    if args.use_ipex:
        torch_index_url = os.environ.get(
            'TORCH_INDEX_URL',
            "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
        )
        torch_command = os.environ.get(
            'TORCH_COMMAND',
            f"uv pip install torch==2.0.0a0 intel-extension-for-pytorch==2.0.110+gitba7f6c1 "
            f"--index-url {torch_index_url}"
        )

    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    requirements_file_for_npu = os.environ.get('REQS_FILE_FOR_NPU', "requirements_npu.txt")

    openclip_package = os.environ.get(
        'OPENCLIP_PACKAGE',
        "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip"
    )

    assets_repo = os.environ.get(
        'ASSETS_REPO',
        "https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git"
    )
    stable_diffusion_repo = os.environ.get(
        'STABLE_DIFFUSION_REPO',
        "https://github.com/joypaul162/Stability-AI-stablediffusion.git"
    )
    stable_diffusion_xl_repo = os.environ.get(
        'STABLE_DIFFUSION_XL_REPO',
        "https://github.com/Stability-AI/generative-models.git"
    )
    k_diffusion_repo = os.environ.get(
        'K_DIFFUSION_REPO',
        'https://github.com/crowsonkb/k-diffusion.git'
    )
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    assets_commit_hash = os.environ.get('ASSETS_COMMIT_HASH', "6f7db241d2f8ba7457bac5ca9753331f0c266917")
    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "f16630a927e00098b524d687640719e4eb469b76")
    stable_diffusion_xl_commit_hash = os.environ.get('STABLE_DIFFUSION_XL_COMMIT_HASH', "45c443b316737a4ab6e40413d7794a7f5657c19f")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "ab527a9a6d347f364e3d185ba6d714e22d80cb3c")
    blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

    # Signal webui.sh that webui needs to be restarted when it stops execution
    try:
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
    except OSError:
        pass

    if not args.skip_python_version_check:
        check_python_version()

    startup_timer.record("checks")

    commit = commit_hash()
    tag = git_tag()
    startup_timer.record("git version info")

    print(f"Python {sys.version}")
    print(f"Version: {tag}")
    print(f"Commit hash: {commit}")

    # Install setuptools first (needed for building packages)
    if not args.skip_install:
        install_setuptools()

    # Install torch BEFORE clip to ensure correct backend (ROCm vs CUDA)
    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(torch_command, "Installing torch and torchvision", "Couldn't install torch", live=True)
        startup_timer.record("install torch")

    # Now install CLIP (after torch, so it uses the correct torch version)
    if not args.skip_install:
        install_clip()

    # Skip CUDA test for AMD, IPEX, or if explicitly requested
    if args.use_ipex or is_amd_gpu():
        args.skip_torch_cuda_test = True

    if not args.skip_torch_cuda_test and not check_run_python("import torch; assert torch.cuda.is_available()"):
        raise RuntimeError(
            'Torch is not able to use GPU; '
            'add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'
        )
    startup_timer.record("torch GPU test")

    # Verify AMD ROCm setup
    if is_amd_gpu() and not args.skip_torch_cuda_test:
        if not check_run_python("import torch; assert torch.cuda.is_available() or hasattr(torch.version, 'hip')"):
            print("Warning: AMD GPU detected but ROCm may not be properly configured")
            print("Make sure ROCm is installed and HSA_OVERRIDE_GFX_VERSION is set if needed")

    if not is_installed("open_clip"):
        run_pip(f"install {openclip_package}", "open_clip")
        startup_timer.record("install open_clip")

    # Only install xformers for NVIDIA GPUs
    if xformers_package is not None and not is_amd_gpu():
        if (not is_installed("xformers") or args.reinstall_xformers) and args.xformers:
            run_pip(f"install --upgrade --reinstall --no-deps {xformers_package}", "xformers")
            startup_timer.record("install xformers")
    elif is_amd_gpu() and is_installed("xformers"):
        # Remove xformers if AMD GPU (not compatible)
        print("AMD GPU detected - removing incompatible xformers...")
        try:
            run("uv pip uninstall xformers -y", "Removing xformers", "Couldn't remove xformers", live=True)
        except Exception:
            pass

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")
        startup_timer.record("install ngrok")

    os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

    git_clone(assets_repo, repo_dir('stable-diffusion-webui-assets'), "assets", assets_commit_hash)
    git_clone(stable_diffusion_repo, repo_dir('stable-diffusion-stability-ai'),
              "Stable Diffusion", stable_diffusion_commit_hash)
    git_clone(stable_diffusion_xl_repo, repo_dir('generative-models'),
              "Stable Diffusion XL", stable_diffusion_xl_commit_hash)
    git_clone(k_diffusion_repo, repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
    git_clone(blip_repo, repo_dir('BLIP'), "BLIP", blip_commit_hash)

    startup_timer.record("clone repositories")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if not requirements_met(requirements_file):
        run_pip(f'install -r "{requirements_file}"', "requirements")
        startup_timer.record("install requirements")

    if not os.path.isfile(requirements_file_for_npu):
        requirements_file_for_npu = os.path.join(script_path, requirements_file_for_npu)

    if "torch_npu" in torch_command and not requirements_met(requirements_file_for_npu):
        run_pip(f'install -r "{requirements_file_for_npu}"', "requirements_for_npu")
        startup_timer.record("install requirements_for_npu")

    if not args.skip_install:
        run_extensions_installers(settings_file=args.ui_settings_file)

    if args.update_check:
        version_check(commit)
        startup_timer.record("check version")

    if args.update_all_extensions:
        git_pull_recursive(extensions_dir)
        startup_timer.record("update extensions")

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        sys.exit(0)


def configure_for_tests():
    test_args = [
        ("--api", None),
        ("--ckpt", os.path.join(script_path, "test/test_files/empty.pt")),
        ("--skip-torch-cuda-test", None),
        ("--disable-nan-check", None),
    ]

    for arg, value in test_args:
        if arg not in sys.argv:
            sys.argv.append(arg)
            if value is not None:
                sys.argv.append(value)

    os.environ['COMMANDLINE_ARGS'] = ""


def configure_forge_reference_checkout(a1111_home: Path):
    """Set model paths based on an existing A1111 checkout."""
    class ModelRef(NamedTuple):
        arg_name: str
        relative_path: str

    refs = [
        ModelRef("--ckpt-dir", "models/Stable-diffusion"),
        ModelRef("--vae-dir", "models/VAE"),
        ModelRef("--hypernetwork-dir", "models/hypernetworks"),
        ModelRef("--embeddings-dir", "embeddings"),
        ModelRef("--lora-dir", "models/Lora"),
        ModelRef("--controlnet-dir", "models/ControlNet"),
        ModelRef("--controlnet-preprocessor-models-dir",
                 "extensions/sd-webui-controlnet/annotator/downloads"),
    ]

    for ref in refs:
        target_path = a1111_home / ref.relative_path
        if not target_path.exists():
            print(f"Path {target_path} does not exist. Skipping {ref.arg_name}")
            continue

        if ref.arg_name in sys.argv:
            continue

        sys.argv.extend([ref.arg_name, str(target_path)])


def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} "
          f"with arguments: {shlex.join(sys.argv[1:])}")

    import webui

    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()

    from modules_forge import main_thread
    main_thread.loop()


def dump_sysinfo():
    from modules import sysinfo
    from datetime import datetime, timezone

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename
