import subprocess
import sys
import os
import shutil
from pathlib import Path

repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"


def get_installed_version(package: str):
    """Get installed package version using importlib.metadata (fast, no pkg_resources)."""
    import importlib.metadata
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        # Try normalized names
        normalized = package.lower().replace('-', '_').replace('.', '_')
        try:
            return importlib.metadata.version(normalized)
        except importlib.metadata.PackageNotFoundError:
            return None


def compare_versions(v1, v2):
    """Compare version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
    import re
    def normalize(v):
        parts = []
        for part in re.split(r'[.\-_]', str(v)):
            match = re.match(r'^(\d+)', part)
            if match:
                parts.append(int(match.group(1)))
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


def find_uv():
    """Find uv executable."""
    # Check if uv is in PATH
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path
    # Common locations
    for path in ["/usr/bin/uv", "/usr/local/bin/uv", os.path.expanduser("~/.cargo/bin/uv")]:
        if os.path.isfile(path):
            return path
    return None


def pip_install(package, desc=None):
    """Install a package using uv pip with timeout."""
    uv = find_uv()
    if not uv:
        print(f"  Warning: uv not found, skipping {package}")
        return False

    try:
        cmd = [uv, "pip", "install", package]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"  Warning: Failed to install {package}: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Warning: Timeout installing {package}")
        return False
    except Exception as e:
        print(f"  Warning: Error installing {package}: {e}")
        return False


def install_requirements(req_file):
    """Install requirements from file."""
    if not os.path.exists(req_file):
        return

    uv = find_uv()
    if not uv:
        print("  Warning: uv not found, skipping requirements")
        return

    packages_to_install = []

    with open(req_file) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                if "==" in line:
                    package_name, package_version = line.split("==", 1)
                    installed = get_installed_version(package_name)
                    if installed != package_version:
                        packages_to_install.append(line)
                elif ">=" in line:
                    package_name, package_version = line.split(">=", 1)
                    installed = get_installed_version(package_name)
                    if not installed or compare_versions(installed, package_version) < 0:
                        packages_to_install.append(line)
                elif line.startswith(('git+', 'http://', 'https://')):
                    # URL-based package
                    packages_to_install.append(line)
                else:
                    # Simple package name
                    package_name = line.split('@')[0].split('[')[0].strip()
                    if not get_installed_version(package_name):
                        packages_to_install.append(line)
            except Exception as e:
                print(f"  Warning: Error parsing requirement {line}: {e}")

    if packages_to_install:
        print(f"  Installing {len(packages_to_install)} packages...")
        try:
            cmd = [uv, "pip", "install"] + packages_to_install
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"  Batch install failed, trying individually...")
                for pkg in packages_to_install:
                    pip_install(pkg)
        except subprocess.TimeoutExpired:
            print(f"  Warning: Timeout during batch install, trying individually...")
            for pkg in packages_to_install:
                pip_install(pkg)
    else:
        print("  All requirements already satisfied.")


def try_install_wheel(pkg_name, wheel_url, min_version=None):
    """Install from wheel if not already installed or version too old."""
    current = get_installed_version(pkg_name)

    if current is not None:
        if min_version is None:
            return
        if compare_versions(current, min_version) >= 0:
            return

    print(f"  Installing {pkg_name}...")
    pip_install(wheel_url)


def try_install_insightface():
    """Install insightface if not present."""
    if get_installed_version("insightface") is not None:
        return

    wheel_url = os.environ.get("INSIGHTFACE_WHEEL", "")

    if wheel_url:
        pip_install(wheel_url)
    else:
        # Try from PyPI
        pip_install("insightface")


def try_remove_legacy_submodule():
    """Remove old submodule directory if exists."""
    submodule = repo_root / "annotator" / "hand_refiner_portable"
    if submodule.exists():
        try:
            shutil.rmtree(submodule)
        except Exception as e:
            print(f"  Warning: Could not remove {submodule}: {e}")


# Main installation
print("forge_legacy_preprocessor: Checking requirements...")

install_requirements(main_req_file)

try_install_insightface()

try_install_wheel(
    "handrefinerportable",
    os.environ.get(
        "HANDREFINER_WHEEL",
        "https://github.com/huchenlei/HandRefinerPortable/releases/download/v1.0.1/handrefinerportable-2024.2.12.0-py2.py3-none-any.whl"
    ),
    min_version="2024.2.12.0"
)

try_install_wheel(
    "depth_anything",
    os.environ.get(
        "DEPTH_ANYTHING_WHEEL",
        "https://github.com/huchenlei/Depth-Anything/releases/download/v1.0.0/depth_anything-2024.1.22.0-py2.py3-none-any.whl"
    )
)

try_remove_legacy_submodule()

print("forge_legacy_preprocessor: Done.")
