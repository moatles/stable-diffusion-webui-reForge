#!/usr/bin/env bash
#################################################
# Optimized install script for stable-diffusion-webui-reForge
# Specifically configured for AMD GPUs with automatic detection
#################################################

set -euo pipefail

# Color codes
readonly RED='\033[1;31m'
readonly GREEN='\033[1;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[1;34m'
readonly CYAN='\033[1;36m'
readonly NC='\033[0m'

# Environment exports for Python
export PYTHONMALLOC="malloc"
export PYTHONDONTWRITEBYTECODE="1"
export PYTHONUNBUFFERED="1"
export PYTHONFAULTHANDLER="1"

# AMD GPU memory settings (optimized for performance)
export GPU_MAX_HEAP_SIZE="95"
export GPU_MAX_ALLOC_PERCENT="95"
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Script directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
readonly SCRIPT_DIR
readonly DELIMITER="################################################################"

# Global state
declare PYTHON_CMD=""
declare GPU_VENDOR=""
declare GPU_MODEL=""
declare GPU_VRAM_MB=0
declare GFX_VERSION=""
declare ROCM_VERSION=""
declare IS_AMD=false
declare IS_NVIDIA=false
declare HAS_INTERNET=false
declare -a EXTRA_ARGS=("--theme=dark")

# Configuration defaults
declare install_dir=""
declare clone_dir=""
declare venv_dir=""
declare LAUNCH_SCRIPT=""
declare can_run_as_root=0
declare first_launch=0

#################################################
# Utility Functions
#################################################

print_delimiter() { printf "\n%s\n" "$DELIMITER"; }
print_info() { printf "${BLUE}[INFO]${NC} %s\n" "$1"; }
print_success() { printf "${GREEN}[OK]${NC} %s\n" "$1"; }
print_warning() { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$1" >&2; }
die() { print_error "$1"; exit 1; }

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local response

    if [[ "$default" == "y" ]]; then
        read -rp "$prompt [Y/n]: " response
        [[ -z "$response" || "$response" =~ ^[Yy] ]]
    else
        read -rp "$prompt [y/N]: " response
        [[ "$response" =~ ^[Yy] ]]
    fi
}

#################################################
# Async Helper - Run commands in parallel where possible
#################################################

declare -A ASYNC_PIDS=()

async_run() {
    local name="$1"
    shift
    "$@" &
    ASYNC_PIDS["$name"]=$!
}

async_wait() {
    local name="$1"
    local pid="${ASYNC_PIDS[$name]:-}"
    if [[ -n "$pid" ]]; then
        wait "$pid" 2>/dev/null || true
        unset "ASYNC_PIDS[$name]"
    fi
}

async_wait_all() {
    for name in "${!ASYNC_PIDS[@]}"; do
        async_wait "$name"
    done
}

#################################################
# Prerequisite Checks
#################################################

check_not_root() {
    if [[ $(id -u) -eq 0 ]] && [[ $can_run_as_root -eq 0 ]]; then
        die "This script must not be launched as root. Use -f flag to override."
    fi
    print_success "Running as user: $(whoami)"
}

check_64bit() {
    [[ $(getconf LONG_BIT) -eq 64 ]] || die "32-bit OS is not supported"
    print_success "64-bit system detected"
}

check_internet() {
    print_info "Checking internet connectivity..."
    # Run connectivity checks in parallel
    local -a check_hosts=("8.8.8.8" "1.1.1.1" "pypi.org")

    for host in "${check_hosts[@]}"; do
        if ping -c 1 -W 2 "$host" &>/dev/null; then
            HAS_INTERNET=true
            print_success "Internet connection available"
            return 0
        fi
    done

    # Try curl as fallback
    if curl -s --max-time 3 https://pypi.org >/dev/null 2>&1; then
        HAS_INTERNET=true
        print_success "Internet connection available"
        return 0
    fi

    HAS_INTERNET=false
    print_warning "No internet connection - will skip remote package checks unless cached"
}

check_uv_installed() {
    print_info "Checking for uv package manager..."
    if ! command -v uv &>/dev/null; then
        print_error "uv is not installed!"
        printf "\n${YELLOW}Install uv with one of these methods:${NC}\n"
        printf "  ${CYAN}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}\n"
        printf "  ${CYAN}yay -S uv${NC}\n"
        printf "  ${CYAN}cargo install uv${NC}\n"
        die "Please install uv and try again"
    fi
    print_success "uv installed: $(uv --version 2>/dev/null | head -1)"
}

check_python310() {
    print_info "Checking for Python 3.10..."

    local -a python_candidates=("python3.10" "python310")

    for cmd in "${python_candidates[@]}"; do
        if command -v "$cmd" &>/dev/null; then
            local version
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
            if [[ "$version" == "3.10" ]]; then
                PYTHON_CMD="$cmd"
                print_success "Python 3.10 found: $("$cmd" --version)"
                return 0
            fi
        fi
    done

    # Check generic python3
    if command -v python3 &>/dev/null; then
        local version
        version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
        if [[ "$version" == "3.10" ]]; then
            PYTHON_CMD="python3"
            print_success "Python 3.10 found: $(python3 --version)"
            return 0
        fi
    fi

    print_error "Python 3.10 is required but not found!"
    printf "\n${YELLOW}Install Python 3.10:${NC}\n"
    printf "\n  ${GREEN}Arch/CachyOS:${NC}\n"
    printf "    ${CYAN}yay -S python310${NC}\n"
    printf "\n  ${GREEN}Ubuntu/Debian:${NC}\n"
    printf "    ${CYAN}sudo add-apt-repository ppa:deadsnakes/ppa${NC}\n"
    printf "    ${CYAN}sudo apt install python3.10 python3.10-venv${NC}\n"
    printf "\n  ${GREEN}Fedora:${NC}\n"
    printf "    ${CYAN}sudo dnf install python3.10${NC}\n"
    die "Please install Python 3.10 and try again"
}

check_git() {
    print_info "Checking for git..."
    command -v git &>/dev/null || die "git is not installed"
    export GIT="${GIT:-git}"
    print_success "git installed"
}

#################################################
# Swap Check
#################################################

check_swap() {
    print_info "Checking swap configuration..."

    local total_swap_kb
    total_swap_kb=$(awk '/SwapTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo "0")
    local total_swap_gb=$((total_swap_kb / 1024 / 1024))

    if [[ $total_swap_gb -lt 32 ]]; then
        print_warning "Swap space is ${total_swap_gb}GB - recommended minimum is 32GB!"
        printf "\n${YELLOW}Large models require significant swap space.${NC}\n"
        printf "${YELLOW}Create a 32GB+ swapfile:${NC}\n"
        printf "  ${CYAN}sudo fallocate -l 32G /swapfile${NC}\n"
        printf "  ${CYAN}sudo chmod 600 /swapfile${NC}\n"
        printf "  ${CYAN}sudo mkswap /swapfile${NC}\n"
        printf "  ${CYAN}sudo swapon /swapfile${NC}\n"
        printf "  ${CYAN}echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab${NC}\n\n"

        # Try to enable existing swapfile
        for swap_file in "/extra/swapfile" "/swapfile"; do
            if [[ -f "$swap_file" ]] && ! grep -q "$swap_file" /proc/swaps 2>/dev/null; then
                if sudo swapon "$swap_file" 2>/dev/null; then
                    print_success "Enabled swapfile: $swap_file"
                    return 0
                fi
            fi
        done
    else
        print_success "Swap space: ${total_swap_gb}GB"
    fi
}

#################################################
# GPU Detection
#################################################

detect_gpu() {
    print_info "Detecting GPU..."

    # Try AMD first
    if detect_amd_gpu; then
        return 0
    fi

    # Try NVIDIA
    if detect_nvidia_gpu; then
        return 0
    fi

    print_warning "No supported GPU detected - will use CPU mode"
}

detect_amd_gpu() {
    # Check for AMD GPU using multiple methods
    local amd_detected=false

    # Method 1: rocm-smi
    if command -v rocm-smi &>/dev/null; then
        local product_name
        product_name=$(rocm-smi --showproductname 2>/dev/null | grep -i "card series" | head -1 | sed 's/.*:\s*//' || true)

        if [[ -n "$product_name" ]]; then
            GPU_MODEL="$product_name"
            amd_detected=true
        fi

        # Get VRAM
        local vram_bytes
        vram_bytes=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i "total memory" | grep -oP '\d+' | head -1 || echo "0")
        if [[ "$vram_bytes" -gt 0 ]]; then
            GPU_VRAM_MB=$((vram_bytes / 1024 / 1024))
        fi
    fi

    # Method 2: lspci fallback
    if [[ "$amd_detected" == false ]] && command -v lspci &>/dev/null; then
        local amd_gpu
        amd_gpu=$(lspci 2>/dev/null | grep -i "vga\|3d" | grep -i "amd\|radeon" | head -1 || true)
        if [[ -n "$amd_gpu" ]]; then
            GPU_MODEL=$(echo "$amd_gpu" | sed 's/.*: //' | sed 's/\[.*\]//')
            amd_detected=true

            # Try to get VRAM from sysfs
            for card_dir in /sys/class/drm/card*/device; do
                if [[ -f "$card_dir/mem_info_vram_total" ]]; then
                    local vram_bytes
                    vram_bytes=$(cat "$card_dir/mem_info_vram_total" 2>/dev/null || echo "0")
                    GPU_VRAM_MB=$((vram_bytes / 1024 / 1024))
                    break
                fi
            done
        fi
    fi

    if [[ "$amd_detected" == true ]]; then
        IS_AMD=true
        GPU_VENDOR="AMD"
        print_success "Detected AMD GPU: $GPU_MODEL"
        [[ $GPU_VRAM_MB -gt 0 ]] && print_info "VRAM: ${GPU_VRAM_MB}MB ($((GPU_VRAM_MB / 1024))GB)"

        detect_gfx_version
        configure_amd_settings
        return 0
    fi

    return 1
}

detect_nvidia_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        return 1
    fi

    local nvidia_info
    nvidia_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)

    if [[ -z "$nvidia_info" ]]; then
        return 1
    fi

    IS_NVIDIA=true
    GPU_VENDOR="NVIDIA"
    GPU_MODEL=$(echo "$nvidia_info" | cut -d',' -f1 | xargs)

    local vram_str
    vram_str=$(echo "$nvidia_info" | cut -d',' -f2 | xargs)
    GPU_VRAM_MB=$(echo "$vram_str" | grep -oP '^\d+' || echo "0")

    print_success "Detected NVIDIA GPU: $GPU_MODEL"
    [[ $GPU_VRAM_MB -gt 0 ]] && print_info "VRAM: ${GPU_VRAM_MB}MB ($((GPU_VRAM_MB / 1024))GB)"

    return 0
}

detect_gfx_version() {
    print_info "Detecting AMD GFX version..."

    # Known GPU to GFX version mappings (RX 6000 = RDNA2, RX 7000 = RDNA3)
    declare -A GFX_MAP=(
        # RDNA2 - RX 6000 series
        ["6500"]="10.3.0"
        ["6600"]="10.3.0"
        ["6650"]="10.3.0"
        ["6700"]="10.3.0"
        ["6750"]="10.3.0"
        ["6800"]="10.3.0"
        ["6900"]="10.3.0"
        ["6950"]="10.3.0"
        # RDNA3 - RX 7000 series
        ["7600"]="11.0.0"
        ["7700"]="11.0.0"
        ["7800"]="11.0.0"
        ["7900"]="11.0.0"
    )

    # Try to match known GPU model
    for gpu_num in "${!GFX_MAP[@]}"; do
        if [[ "$GPU_MODEL" == *"$gpu_num"* ]]; then
            GFX_VERSION="${GFX_MAP[$gpu_num]}"
            print_success "Matched GFX version: $GFX_VERSION (for *$gpu_num* GPU)"
            export HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION"
            return 0
        fi
    done

    # Try rocminfo for automatic detection
    if command -v rocminfo &>/dev/null; then
        local gfx_raw
        gfx_raw=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || true)

        if [[ -n "$gfx_raw" ]]; then
            local gfx_num="${gfx_raw#gfx}"

            # Convert gfxXXXX to X.X.X format
            case "$gfx_num" in
                1030|1031|1032) GFX_VERSION="10.3.0" ;;
                1033) GFX_VERSION="10.3.3" ;;
                1034) GFX_VERSION="10.3.4" ;;
                1035) GFX_VERSION="10.3.5" ;;
                1100) GFX_VERSION="11.0.0" ;;
                1101) GFX_VERSION="11.0.1" ;;
                1102) GFX_VERSION="11.0.2" ;;
                1103) GFX_VERSION="11.0.3" ;;
                900)  GFX_VERSION="9.0.0" ;;
                906)  GFX_VERSION="9.0.6" ;;
                908)  GFX_VERSION="9.0.8" ;;
                90a)  GFX_VERSION="9.0.10" ;;
                *)
                    # Generic conversion: gfx1030 -> 10.3.0
                    if [[ ${#gfx_num} -ge 3 ]]; then
                        local major="${gfx_num:0:${#gfx_num}-2}"
                        local minor="${gfx_num: -2:1}"
                        local patch="${gfx_num: -1}"
                        GFX_VERSION="${major}.${minor}.${patch}"
                    else
                        GFX_VERSION="10.3.0"
                    fi
                    ;;
            esac

            print_success "Auto-detected GFX version from rocminfo: $GFX_VERSION ($gfx_raw)"
            export HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION"
            return 0
        fi
    fi

    # Fallback
    print_warning "Could not auto-detect GFX version, using default 10.3.0"
    GFX_VERSION="10.3.0"
    export HSA_OVERRIDE_GFX_VERSION="$GFX_VERSION"
}

get_gpu_generation() {
    # Returns: "rdna2" for RX 6000 series, "rdna3" for RX 7000 series
    case "$GPU_MODEL" in
        *6[0-9][0-9][0-9]*) echo "rdna2" ;;
        *7[0-9][0-9][0-9]*) echo "rdna3" ;;
        *) echo "unknown" ;;
    esac
}

configure_amd_settings() {
    print_info "Configuring AMD-specific settings..."

    # AMD cards must disable xformers
    EXTRA_ARGS+=("--disable-xformers")

    # AMD cards should use FP16 mode
    EXTRA_ARGS+=("--all-in-fp16" "--no-half-vae")

    # Low VRAM mode for 12GB and under
    if [[ $GPU_VRAM_MB -gt 0 ]] && [[ $GPU_VRAM_MB -le 12288 ]]; then
        EXTRA_ARGS+=("--always-no-vram")
        print_info "Enabled low VRAM mode (${GPU_VRAM_MB}MB <= 12GB)"
    fi

    print_success "AMD settings configured: ${EXTRA_ARGS[*]}"
}

#################################################
# ROCm Management
#################################################

get_installed_torch_rocm() {
    # Check if torch is installed and get ROCm version
    local venv_python="${venv_dir:-venv}/bin/python"

    if [[ ! -x "$venv_python" ]]; then
        echo ""
        return
    fi

    local hip_version
    hip_version=$("$venv_python" -c "
import torch
v = getattr(torch.version, 'hip', None)
if v:
    print(v.split('.')[0] + '.' + v.split('.')[1] if '.' in v else v)
" 2>/dev/null || true)

    echo "$hip_version"
}

select_rocm_version() {
    if [[ "$IS_AMD" != true ]]; then
        return 0
    fi

    print_info "Checking ROCm/PyTorch configuration..."

    local installed_rocm
    installed_rocm=$(get_installed_torch_rocm)

    if [[ -n "$installed_rocm" ]]; then
        print_success "ROCm PyTorch already installed (version: $installed_rocm)"
        ROCM_VERSION="$installed_rocm"
        return 0
    fi

    local gpu_gen
    gpu_gen=$(get_gpu_generation)

    printf "\n${CYAN}═══════════════════════════════════════════════════════════${NC}\n"
    printf "${CYAN}                    ROCm Version Selection${NC}\n"
    printf "${CYAN}═══════════════════════════════════════════════════════════${NC}\n\n"

    case "$gpu_gen" in
        rdna2)
            printf "${YELLOW}RX 6000 series (RDNA2) detected${NC}\n"
            printf "  → Recommended: ROCm 6.3 (nightly)\n\n"
            ;;
        rdna3)
            printf "${YELLOW}RX 7000 series (RDNA3) detected${NC}\n"
            printf "  → ROCm 6.3 or 7.1 both work\n\n"
            ;;
        *)
            printf "${YELLOW}GPU generation unknown${NC}\n"
            printf "  → Try ROCm 7.1 (latest stable)\n\n"
            ;;
    esac

    printf "Available options:\n"
    printf "  ${GREEN}1)${NC} ROCm 6.3 (nightly) - Best compatibility for RX 6000/7000\n"
    printf "  ${GREEN}2)${NC} ROCm 7.1 (stable)  - Latest stable release\n"
    printf "  ${GREEN}3)${NC} Skip torch installation\n\n"

    local choice
    read -rp "Select ROCm version [1-3] (default: 1): " choice
    choice="${choice:-1}"

    local force_reinstall=""
    if [[ -n "$installed_rocm" ]]; then
        force_reinstall="--reinstall"
        print_info "Will force reinstall to change ROCm version"
    fi

    case "$choice" in
        1)
            ROCM_VERSION="6.3"
            export TORCH_COMMAND="uv pip install $force_reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3"
            ;;
        2)
            ROCM_VERSION="7.1"
            export TORCH_COMMAND="uv pip install $force_reinstall torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1"
            ;;
        3)
            print_info "Skipping torch installation"
            return 0
            ;;
        *)
            ROCM_VERSION="6.3"
            export TORCH_COMMAND="uv pip install $force_reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.3"
            ;;
    esac

    print_success "Selected ROCm version: $ROCM_VERSION"
}

uninstall_xformers_if_amd() {
    if [[ "$IS_AMD" != true ]]; then
        return 0
    fi

    local venv_python="${venv_dir:-venv}/bin/python"

    if [[ ! -x "$venv_python" ]]; then
        return 0
    fi

    if "$venv_python" -c "import xformers" &>/dev/null; then
        print_info "AMD GPU detected - uninstalling incompatible xformers..."
        uv pip uninstall xformers -y 2>/dev/null || true
        print_success "xformers removed"
    fi
}

#################################################
# Virtual Environment Setup
#################################################

setup_venv() {
    if [[ "${venv_dir}" == "-" ]]; then
        print_info "Virtual environment disabled"
        return 0
    fi

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        print_info "Already in virtual environment: $VIRTUAL_ENV"
        PYTHON_CMD="${VIRTUAL_ENV}/bin/python"
        return 0
    fi

    print_info "Setting up Python 3.10 virtual environment..."

    cd "${install_dir}/${clone_dir}" || die "Cannot cd to ${install_dir}/${clone_dir}"

    if [[ ! -d "${venv_dir}" ]]; then
        print_info "Creating new venv..."
        uv venv "${venv_dir}" --python=python3.10 || die "Failed to create venv with uv"
        first_launch=1
        print_success "Virtual environment created"
    fi

    if [[ -f "${venv_dir}/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "${venv_dir}/bin/activate"
        PYTHON_CMD="${venv_dir}/bin/python"
        print_success "Virtual environment activated"
    else
        die "Cannot find venv activation script"
    fi
}

#################################################
# TCMalloc Setup
#################################################

setup_tcmalloc() {
    if [[ -n "${NO_TCMALLOC:-}" ]]; then
        return 0
    fi

    # Skip if already set
    if [[ -n "${LD_PRELOAD:-}" ]] && [[ "$LD_PRELOAD" == *"tcmalloc"* ]]; then
        print_success "TCMalloc already configured: $LD_PRELOAD"
        return 0
    fi

    local -a tcmalloc_paths=(
        "/usr/lib/libtcmalloc.so"
        "/usr/lib/libtcmalloc_minimal.so"
        "/usr/lib64/libtcmalloc.so"
        "/usr/lib64/libtcmalloc_minimal.so"
        "/usr/lib/x86_64-linux-gnu/libtcmalloc.so"
        "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so"
    )

    for lib_path in "${tcmalloc_paths[@]}"; do
        if [[ -f "$lib_path" ]]; then
            export LD_PRELOAD="$lib_path"
            print_success "TCMalloc loaded: $lib_path"
            return 0
        fi
    done

    # Try ldconfig
    local tcmalloc_lib
    tcmalloc_lib=$(ldconfig -p 2>/dev/null | grep -oP '/\S+libtcmalloc[^/]*\.so\S*' | head -1 || true)

    if [[ -n "$tcmalloc_lib" ]]; then
        export LD_PRELOAD="$tcmalloc_lib"
        print_success "TCMalloc loaded: $tcmalloc_lib"
        return 0
    fi

    print_warning "TCMalloc not found - install google-perftools for better memory performance"
}

#################################################
# Cache Check for Offline Mode
#################################################

check_packages_cached() {
    # Check if essential packages are in uv cache
    if [[ "$HAS_INTERNET" == true ]]; then
        return 0  # Has internet, no need to check cache
    fi

    print_info "Checking package cache for offline mode..."

    # uv caches packages in ~/.cache/uv
    local uv_cache="${HOME}/.cache/uv"

    if [[ -d "$uv_cache" ]] && [[ $(find "$uv_cache" -name "*.whl" 2>/dev/null | wc -l) -gt 0 ]]; then
        print_success "Package cache found - can install cached packages"
        return 0
    fi

    print_warning "No package cache found - some installations may fail without internet"
    return 1
}

#################################################
# Main Installation
#################################################

main() {
    print_delimiter
    printf "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}\n"
    printf "${GREEN}║   Stable Diffusion WebUI reForge - Installation Script    ║${NC}\n"
    printf "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}\n"
    print_delimiter

    # Parse arguments
    while getopts "f" flag 2>/dev/null; do
        case ${flag} in
            f) can_run_as_root=1 ;;
            *) break ;;
        esac
    done
    shift $((OPTIND - 1))

    # Run prerequisite checks (some in parallel)
    check_not_root
    check_64bit

    # Parallel checks
    async_run "internet" check_internet
    async_run "swap" check_swap

    # Sequential required checks
    check_uv_installed
    check_python310
    check_git

    # Wait for parallel checks
    async_wait_all

    # Check cache if offline
    check_packages_cached

    # GPU detection and configuration
    detect_gpu

    # Load user configuration
    if [[ -f "$SCRIPT_DIR/webui-user.sh" ]]; then
        print_info "Loading user configuration..."
        # shellcheck source=/dev/null
        source "$SCRIPT_DIR/webui-user.sh"
    fi

    # Set defaults
    install_dir="${install_dir:-$SCRIPT_DIR}"
    clone_dir="${clone_dir:-stable-diffusion-webui}"
    venv_dir="${venv_dir:-venv}"
    LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-launch.py}"

    # Handle existing git repo
    if [[ -d "$SCRIPT_DIR/.git" ]]; then
        print_info "Using existing repo as install directory"
        install_dir="${SCRIPT_DIR}/../"
        clone_dir="${SCRIPT_DIR##*/}"
    fi

    # Navigate to install directory
    cd "${install_dir}" || die "Cannot cd to ${install_dir}"

    # Clone if needed
    if [[ ! -d "${clone_dir}" ]]; then
        if [[ "$HAS_INTERNET" == false ]]; then
            die "Cannot clone repository without internet connection"
        fi
        print_info "Cloning stable-diffusion-webui..."
        "${GIT}" clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "${clone_dir}"
    fi

    cd "${clone_dir}" || die "Cannot cd to ${clone_dir}"

    # Setup virtual environment
    setup_venv

    # AMD-specific setup
    if [[ "$IS_AMD" == true ]]; then
        select_rocm_version
        uninstall_xformers_if_amd
    fi

    # Setup TCMalloc
    setup_tcmalloc

    # Configure NVIDIA if detected
    if [[ "$IS_NVIDIA" == true ]]; then
        # Low VRAM mode for 12GB and under
        if [[ $GPU_VRAM_MB -gt 0 ]] && [[ $GPU_VRAM_MB -le 12288 ]]; then
            EXTRA_ARGS+=("--always-no-vram")
            print_info "Enabled low VRAM mode (${GPU_VRAM_MB}MB <= 12GB)"
        fi
    fi

    # Environment setup
    export SD_WEBUI_RESTART=tmp/restart
    export ERROR_REPORTING=FALSE
    export PIP_IGNORE_INSTALLED=0

    # Build final argument list
    local args_str="${EXTRA_ARGS[*]}"

    print_delimiter
    printf "${GREEN}Configuration Summary${NC}\n"
    print_delimiter
    printf "  GPU: ${CYAN}%s${NC} (%s)\n" "$GPU_MODEL" "$GPU_VENDOR"
    [[ $GPU_VRAM_MB -gt 0 ]] && printf "  VRAM: ${CYAN}%dMB${NC}\n" "$GPU_VRAM_MB"
    [[ -n "$GFX_VERSION" ]] && printf "  GFX Version: ${CYAN}%s${NC}\n" "$GFX_VERSION"
    [[ -n "$ROCM_VERSION" ]] && printf "  ROCm Version: ${CYAN}%s${NC}\n" "$ROCM_VERSION"
    printf "  Python: ${CYAN}%s${NC}\n" "$($PYTHON_CMD --version)"
    printf "  Extra Args: ${CYAN}%s${NC}\n" "$args_str"
    print_delimiter

    printf "\n${GREEN}Launching WebUI...${NC}\n\n"

    # Main loop
    while true; do
        if [[ -n "${ACCELERATE:-}" ]] && [[ "${ACCELERATE}" == "True" ]] && command -v accelerate &>/dev/null; then
            print_info "Using accelerate launcher..."
            accelerate launch --num_cpu_threads_per_process=6 "${LAUNCH_SCRIPT}" "$@" "${EXTRA_ARGS[@]}"
        else
            "${PYTHON_CMD}" -u "${LAUNCH_SCRIPT}" "$@" "${EXTRA_ARGS[@]}"
        fi

        if [[ ! -f tmp/restart ]]; then
            break
        fi

        print_info "Restarting WebUI..."
        sleep 1
    done

    print_success "WebUI stopped"
}

# Entry point
main "$@"
