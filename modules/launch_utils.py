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
import tempfile
from functools import lru_cache
from typing import NamedTuple, Optional, Tuple
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

# Packages that pull in CUDA dependencies and should be installed with --no-deps on AMD
AMD_NODEPS_PACKAGES = {'transformers', 'accelerate', 'diffusers', 'xformers'}

# GPU Detection Cache
_gpu_info_cache: Optional[dict] = None

# Extension installer timeout in seconds
EXTENSION_INSTALL_TIMEOUT = 300

# Cache for requirements check to avoid reinstalling on every launch
_requirements_checked = False


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
            encoding='utf8', stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except Exception:
        return "<none>"


@lru_cache()
def git_tag_a1111():
    try:
        return subprocess.check_output(
            [git, "-C", script_path, "describe", "--tags"],
            encoding='utf8', stderr=subprocess.DEVNULL, timeout=10
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


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live, timeout=None) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if timeout is not None:
        run_kwargs["timeout"] = timeout

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    try:
        result = subprocess.run(**run_kwargs)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s: {command}")

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
    """Check if a package is installed using importlib.metadata only."""
    try:
        importlib.metadata.distribution(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        # Normalize package name and try again
        normalized = package.lower().replace('-', '_').replace('.', '_')
        try:
            importlib.metadata.distribution(normalized)
            return True
        except importlib.metadata.PackageNotFoundError:
            # Try with hyphens
            hyphenated = package.lower().replace('_', '-')
            try:
                importlib.metadata.distribution(hyphenated)
                return True
            except importlib.metadata.PackageNotFoundError:
                return False


def get_package_version(package):
    """Get installed version of a package, or None if not installed."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        # Try normalized name
        normalized = package.lower().replace('-', '_').replace('.', '_')
        try:
            return importlib.metadata.version(normalized)
        except importlib.metadata.PackageNotFoundError:
            # Try with hyphens
            hyphenated = package.lower().replace('_', '-')
            try:
                return importlib.metadata.version(hyphenated)
            except importlib.metadata.PackageNotFoundError:
                return None


def parse_version(v: str) -> Tuple[int, ...]:
    """Parse version string into tuple of integers for comparison."""
    parts = []
    for part in re.split(r'[.\-_+]', str(v)):
        match = re.match(r'^(\d+)', part)
        if match:
            parts.append(int(match.group(1)))
    return tuple(parts) if parts else (0,)


def compare_versions(v1, v2):
    """
    Compare two version strings.
    Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    v1_parts = parse_version(v1)
    v2_parts = parse_version(v2)

    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts = v1_parts + (0,) * (max_len - len(v1_parts))
    v2_parts = v2_parts + (0,) * (max_len - len(v2_parts))

    for a, b in zip(v1_parts, v2_parts):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def version_satisfies(installed: str, requirement: str) -> bool:
    """
    Check if installed version satisfies a version requirement.
    Handles ==, >=, <=, >, <, != operators and comma-separated constraints.
    """
    if not installed or not requirement:
        return not requirement  # No requirement means satisfied

    installed = installed.strip()
    requirement = requirement.strip()

    # Handle comma-separated constraints (e.g., ">=0.21,<0.22")
    if ',' in requirement:
        constraints = [c.strip() for c in requirement.split(',')]
        return all(version_satisfies(installed, c) for c in constraints)

    # Parse operator and version
    match = re.match(r'^([<>=!]+)?\s*(.+)$', requirement)
    if not match:
        return True

    op = match.group(1) or '=='
    required_ver = match.group(2).strip()

    cmp = compare_versions(installed, required_ver)

    if op == '==':
        return cmp == 0
    elif op == '>=':
        return cmp >= 0
    elif op == '<=':
        return cmp <= 0
    elif op == '>':
        return cmp > 0
    elif op == '<':
        return cmp < 0
    elif op == '!=':
        return cmp != 0
    else:
        return True


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live, timeout=120):
    """Run a uv pip command (named run_pip for backward compatibility)."""
    if args.skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url else ''
    return run(
        f'uv pip {command}{index_url_line}',
        desc=f"Installing {desc}" if desc else None,
        errdesc=f"Couldn't install {desc}" if desc else "Couldn't run pip command",
        live=live,
        timeout=timeout
    )


# Alias for explicit uv usage
run_uv = run_pip


def check_run_python(code: str) -> bool:
    try:
        result = subprocess.run([python, "-c", code], capture_output=True, timeout=30)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def git_fix_workspace(directory, name):
    run(f'{git} -C "{directory}" fetch --refetch --no-auto-gc',
        f"Fetching all contents for {name}",
        f"Couldn't fetch {name}", live=True, timeout=300)
    run(f'{git} -C "{directory}" gc --aggressive --prune=now',
        f"Pruning {name}",
        f"Couldn't prune {name}", live=True, timeout=300)


def run_git(directory, name, command, desc=None, errdesc=None, custom_env=None,
            live: bool = default_command_live, autofix=True):
    try:
        return run(f'{git} -C "{directory}" {command}', desc=desc, errdesc=errdesc,
                   custom_env=custom_env, live=live, timeout=120)
    except RuntimeError:
        if not autofix:
            raise

    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(directory, name)
    return run(f'{git} -C "{directory}" {command}', desc=desc, errdesc=errdesc,
               custom_env=custom_env, live=live, timeout=120)


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
            f"Couldn't clone {name}", live=True, timeout=600)
    except RuntimeError:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    if commithash is not None:
        run(f'{git} -C "{directory}" checkout {commithash}', None,
            f"Couldn't checkout {name}'s hash: {commithash}", timeout=60)


def git_pull_recursive(directory):
    for subdir, _, _ in os.walk(directory):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'], timeout=60)
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.TimeoutExpired:
                print(f"Timeout pulling repository in '{subdir}'")
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
    """Run extension installer with timeout protection."""
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    extension_name = os.path.basename(extension_dir)
    print(f"    Running install.py for {extension_name}...")

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        # Use subprocess with timeout instead of exec
        result = subprocess.run(
            [python, path_installer],
            env=env,
            capture_output=True,
            text=True,
            timeout=EXTENSION_INSTALL_TIMEOUT,
            cwd=extension_dir
        )

        if result.stdout:
            output = result.stdout.strip()
            if output:
                for line in output.split('\n'):
                    print(f"    {line}")

        if result.returncode != 0 and result.stderr:
            print(f"    Warning: {result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        print(f"    Warning: install.py for {extension_name} timed out after {EXTENSION_INSTALL_TIMEOUT}s")
    except Exception as e:
        print(f"    Warning: Error running install.py for {extension_name}: {e}")


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
    print("Running extension installers...")

    extensions_count = 0
    builtin_count = 0

    if os.path.isdir(extensions_dir):
        extensions = list_extensions(settings_file)
        extensions_count = len(extensions)
        print(f"  Found {extensions_count} extensions")

        with startup_timer.subcategory("run extensions installers"):
            for i, dirname_extension in enumerate(extensions, 1):
                print(f"  [{i}/{extensions_count}] Processing {dirname_extension}...")
                logging.debug(f"Installing {dirname_extension}")
                path = os.path.join(extensions_dir, dirname_extension)
                if os.path.isdir(path):
                    run_extension_installer(path)
                    startup_timer.record(dirname_extension)

    if os.path.isdir(extensions_builtin_dir):
        extensions_builtin = list_extensions_builtin(settings_file)
        builtin_count = len(extensions_builtin)
        print(f"  Found {builtin_count} builtin extensions")

        with startup_timer.subcategory("run extensions_builtin installers"):
            for i, dirname_extension in enumerate(extensions_builtin, 1):
                print(f"  [{i}/{builtin_count}] Processing builtin {dirname_extension}...")
                logging.debug(f"Installing {dirname_extension}")
                path = os.path.join(extensions_builtin_dir, dirname_extension)
                if os.path.isdir(path):
                    run_extension_installer(path)
                    startup_timer.record(dirname_extension)

    print(f"Extension installers completed. ({extensions_count} extensions, {builtin_count} builtin)")


RE_REQUIREMENT = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:([<>=!]+)\s*([-+_.a-zA-Z0-9]+))?\s*")


def parse_requirement_line(line):
    """
    Parse a requirement line and return (package_name, version_spec, original_line).
    Returns (None, None, None) for comments/empty lines.
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None, None

    # Skip git/url requirements
    if line.startswith(('git+', 'http://', 'https://')):
        return None, None, line

    # Extract package name (handles ==, >=, <=, ~=, !=, etc.)
    match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)\s*([<>=!~,\d\.\s]+)?$', line)
    if match:
        package = match.group(1).split('[')[0]  # Remove extras like [cuda]
        version_spec = match.group(2) or ''
        return package.lower().replace('-', '_'), version_spec.strip(), line

    return None, None, line


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

                # Skip URL-based requirements
                if line.startswith(('git+', 'http://', 'https://')):
                    continue

                # Parse package name and version spec
                match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', line)
                if not match:
                    continue

                package = match.group(1).strip()
                version_spec = match.group(2).strip()

                installed_version = get_package_version(package)
                if installed_version is None:
                    return False

                # Check version constraint if specified
                if version_spec and not version_satisfies(installed_version, version_spec):
                    return False
    except FileNotFoundError:
        return False

    return True


def check_amd_deps_installed() -> bool:
    """
    Check if AMD-specific dependencies are correctly installed.
    Returns True if all deps are satisfied, False if reinstall needed.
    """
    # Check transformers deps
    tokenizers_ver = get_package_version('tokenizers')
    if tokenizers_ver and not version_satisfies(tokenizers_ver, '>=0.21,<0.22'):
        return False

    # Check that essential packages are installed
    essential = ['transformers', 'accelerate', 'huggingface-hub', 'safetensors', 'tokenizers']
    for pkg in essential:
        if not is_installed(pkg):
            return False

    return True


def install_requirements_amd(requirements_file):
    """
    Install requirements for AMD GPU, handling packages that might pull CUDA deps.
    Installs problematic packages with --no-deps to prevent CUDA contamination,
    but first installs their safe dependencies with correct versions.
    """
    global _requirements_checked

    # Skip if already checked this session and deps are satisfied
    if _requirements_checked and check_amd_deps_installed():
        print("  All AMD requirements already satisfied.")
        return

    print(f"Installing requirements for AMD GPU from {requirements_file}")

    # Dependencies of packages that are safe (don't pull CUDA)
    # Version constraints match transformers==4.44.0 and other package requirements
    AMD_SAFE_DEPS = {
        'transformers': [
            'tokenizers>=0.21,<0.22',
            'huggingface-hub>=0.23.0,<1.0',
            'safetensors>=0.4.1',
            'regex!=2019.12.17',
            'requests',
            'tqdm>=4.27',
            'pyyaml>=5.1',
            'packaging>=20.0',
            'filelock',
            'numpy',
        ],
        'accelerate': [
            'huggingface-hub>=0.21.0',
            'safetensors>=0.4.3',
            'pyyaml',
            'packaging>=20.0',
            'psutil',
            'numpy>=1.17',
        ],
        'diffusers': [
            'huggingface-hub>=0.23.2',
            'safetensors>=0.3.1',
            'requests',
            'pillow',
            'pyyaml',
            'packaging',
            'regex!=2019.12.17',
            'importlib-metadata',
            'numpy',
        ],
    }

    try:
        with open(requirements_file, "r", encoding="utf8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Requirements file not found: {requirements_file}")
        return

    # Separate packages into normal and no-deps categories
    normal_packages = []
    nodeps_packages = []
    skip_packages = {'torch', 'torchvision', 'xformers', 'open_clip_torch', 'open-clip-torch'}

    for line in lines:
        package_name, version_spec, original = parse_requirement_line(line)

        if original is None:
            continue  # Skip empty/comment lines

        if package_name is None:
            # URL-based requirement, install as-is
            normal_packages.append(original)
            continue

        if package_name in skip_packages or package_name.replace('_', '-') in skip_packages:
            print(f"  Skipping {package_name} (already handled)")
            continue

        if package_name in AMD_NODEPS_PACKAGES:
            nodeps_packages.append((package_name, original))
        else:
            normal_packages.append(original)

    # Check if normal packages need installation
    normal_needed = []
    for pkg_line in normal_packages:
        pkg_name, ver_spec, _ = parse_requirement_line(pkg_line)
        if pkg_name:
            installed = get_package_version(pkg_name)
            if not installed:
                normal_needed.append(pkg_line)
            elif ver_spec and not version_satisfies(installed, ver_spec):
                normal_needed.append(pkg_line)
        else:
            # URL package - check by other means or just include
            normal_needed.append(pkg_line)

    # Install normal packages if needed
    if normal_needed:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf8') as tmp:
            tmp.write('\n'.join(normal_needed))
            tmp_path = tmp.name

        try:
            run(
                f'uv pip install -r "{tmp_path}"',
                desc=f"Installing {len(normal_needed)} standard requirements",
                errdesc="Couldn't install requirements",
                live=True,
                timeout=600
            )
        finally:
            os.unlink(tmp_path)
    else:
        print("  Standard requirements already satisfied.")

    # Install no-deps packages: first their dependencies, then the package itself
    for package_name, pkg_spec in nodeps_packages:
        # Check if deps need installation
        if package_name in AMD_SAFE_DEPS:
            deps_needed = []
            for dep in AMD_SAFE_DEPS[package_name]:
                # Parse dep name and version
                dep_match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', dep)
                if dep_match:
                    dep_name = dep_match.group(1)
                    dep_ver = dep_match.group(2).strip()
                    installed = get_package_version(dep_name)
                    if not installed:
                        deps_needed.append(dep)
                    elif dep_ver and not version_satisfies(installed, dep_ver):
                        deps_needed.append(dep)

            if deps_needed:
                print(f"  Installing {package_name} dependencies: {', '.join(d.split('>=')[0].split('<')[0] for d in deps_needed)}")
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf8') as tmp:
                        tmp.write('\n'.join(deps_needed))
                        tmp_path = tmp.name
                    try:
                        run(
                            f'uv pip install -r "{tmp_path}"',
                            errdesc=f"Couldn't install {package_name} dependencies",
                            live=True,
                            timeout=180
                        )
                    finally:
                        os.unlink(tmp_path)
                except RuntimeError as e:
                    print(f"  Warning: Failed to install some dependencies: {e}")

        # Then install the package itself with --no-deps if not installed
        if is_installed(package_name):
            print(f"  {package_name} already installed")
        else:
            print(f"  Installing {pkg_spec} with --no-deps (AMD compatibility)")
            try:
                run(
                    f'uv pip install --no-deps "{pkg_spec}"',
                    errdesc=f"Couldn't install {pkg_spec}",
                    live=True,
                    timeout=120
                )
            except RuntimeError as e:
                print(f"  Warning: Failed to install {pkg_spec}: {e}")

    _requirements_checked = True


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
            text=True, stderr=subprocess.DEVNULL, timeout=10
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
            live=True,
            timeout=60
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
        live=True,
        timeout=120
    )
    startup_timer.record("install clip")


def cleanup_nvidia_packages():
    """Remove NVIDIA/CUDA packages that may have been accidentally installed on AMD."""
    nvidia_packages = [
        'nvidia-cuda-cupti-cu12',
        'nvidia-cuda-nvrtc-cu12',
        'nvidia-cuda-runtime-cu12',
        'nvidia-cudnn-cu12',
        'nvidia-cufft-cu12',
        'nvidia-curand-cu12',
        'nvidia-cusolver-cu12',
        'nvidia-cusparse-cu12',
        'nvidia-cusparselt-cu12',
        'nvidia-nccl-cu12',
        'nvidia-nvjitlink-cu12',
        'nvidia-nvtx-cu12',
        'nvidia-cublas-cu12',
        'nvidia-nvshmem-cu12',
        'triton',  # NVIDIA triton (we have pytorch-triton-rocm instead)
    ]

    packages_to_remove = []
    for pkg in nvidia_packages:
        if is_installed(pkg):
            packages_to_remove.append(pkg)

    if not packages_to_remove:
        print("No NVIDIA packages to clean up.")
        return

    print(f"Cleaning up NVIDIA packages on AMD system: {', '.join(packages_to_remove)}")
    try:
        run(
            f"uv pip uninstall {' '.join(packages_to_remove)}",
            desc="Removing NVIDIA packages",
            errdesc="Couldn't remove NVIDIA packages",
            live=True,
            timeout=60
        )
    except Exception as e:
        print(f"Warning: Failed to remove some NVIDIA packages: {e}")


def prepare_environment():
    global _requirements_checked

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
        "https://github.com/moatles/generative-models.git"
    )
    k_diffusion_repo = os.environ.get(
        'K_DIFFUSION_REPO',
        'https://github.com/crowsonkb/k-diffusion.git'
    )
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    assets_commit_hash = os.environ.get('ASSETS_COMMIT_HASH', "6f7db241d2f8ba7457bac5ca9753331f0c266917")
    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "f16630a927e00098b524d687640719e4eb469b76")
    stable_diffusion_xl_commit_hash = os.environ.get('STABLE_DIFFUSION_XL_COMMIT_HASH', "")
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
        run(torch_command, "Installing torch and torchvision", "Couldn't install torch", live=True, timeout=1800)
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
    if is_amd_gpu():
        print("Verifying AMD ROCm setup...")
        if check_run_python("import torch; print(f'ROCm: {torch.version.hip}'); assert torch.cuda.is_available()"):
            print("  ROCm is working correctly.")
        else:
            print("  Warning: AMD GPU detected but ROCm may not be properly configured")
            print("  Make sure ROCm is installed and HSA_OVERRIDE_GFX_VERSION is set if needed")

    # Install open_clip with dependencies (ftfy, regex, tqdm - all safe, no CUDA)
    if not is_installed("open_clip_torch") and not is_installed("open-clip-torch"):
        run_pip(f"install {openclip_package}", "open_clip", timeout=120)
        startup_timer.record("install open_clip")
    else:
        # Ensure open_clip dependencies are present
        openclip_deps = ['ftfy', 'regex', 'tqdm']
        missing_deps = [d for d in openclip_deps if not is_installed(d)]
        if missing_deps:
            run_pip(f"install {' '.join(missing_deps)}", "open_clip dependencies", timeout=60)

    # Only install xformers for NVIDIA GPUs
    if xformers_package is not None and not is_amd_gpu():
        if (not is_installed("xformers") or args.reinstall_xformers) and args.xformers:
            run_pip(f"install --upgrade --reinstall --no-deps {xformers_package}", "xformers", timeout=120)
            startup_timer.record("install xformers")
    elif is_amd_gpu() and is_installed("xformers"):
        # Remove xformers if AMD GPU (not compatible)
        print("AMD GPU detected - removing incompatible xformers...")
        try:
            run("uv pip uninstall xformers", "Removing xformers", "Couldn't remove xformers", live=True, timeout=30)
        except Exception:
            pass

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok", timeout=60)
        startup_timer.record("install ngrok")

    os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

    print("Cloning repositories...")
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

    if is_amd_gpu():
        # Use AMD-specific installer with smart caching
        install_requirements_amd(requirements_file)
    elif not requirements_met(requirements_file):
        run_pip(f'install -r "{requirements_file}"', "requirements", timeout=600)
    startup_timer.record("install requirements")

    if not os.path.isfile(requirements_file_for_npu):
        requirements_file_for_npu = os.path.join(script_path, requirements_file_for_npu)

    if "torch_npu" in torch_command and not requirements_met(requirements_file_for_npu):
        run_pip(f'install -r "{requirements_file_for_npu}"', "requirements_for_npu", timeout=300)
        startup_timer.record("install requirements_for_npu")

    # Clean up any accidentally installed NVIDIA packages on AMD (only first run)
    if is_amd_gpu() and not _requirements_checked:
        cleanup_nvidia_packages()

    print("Running extension installers...")
    if not args.skip_install:
        run_extensions_installers(settings_file=args.ui_settings_file)

    print("Checking for updates...")
    if args.update_check:
        version_check(commit)
        startup_timer.record("check version")

    if args.update_all_extensions:
        git_pull_recursive(extensions_dir)
        startup_timer.record("update extensions")

    print("Environment preparation complete.")

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
