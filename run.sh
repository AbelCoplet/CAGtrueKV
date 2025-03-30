#!/bin/bash

# Set up core environment
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# === METAL ACCELERATION CONFIGURATION (Comprehensive) ===
# Enable Metal performance optimizations for Apple Silicon
export GGML_METAL_DEBUG=0                    # Disable debug mode for performance
export METAL_DEBUG_ERROR_MODE=0              # Minimize error checking overhead
export GGML_METAL_PATH_RESOURCES="$(pwd)/metal" # Metal shader resources path
export METAL_DEVICE_WRAPPER_TYPE=1           # Use the default device
export PYTORCH_ENABLE_MPS_FALLBACK=1         # Enable MPS fallback for unsupported ops
export GGML_METAL_NDEBUG=1                   # Disable debug assertions
export GGML_METAL_FULL_BACKEND=1             # Use full Metal backend features

# Optional: Memory management tuning (Example - can be overridden by UI settings later)
# export GGML_METAL_MEM_MB=5120                # Allocate 5GB for Metal (adjust based on usage)

# Optional: CPU acceleration to complement GPU (Example)
# export GGML_SCHED=METAL_CPU                  # Use both Metal and CPU
# export GGML_METAL_RNGSEED=42                 # Fixed random seed for reproducibility

# Ensure logs directory exists
mkdir -p ~/.llamacag/logs

# === Metal Shader Resources Setup ===
# Create metal directory if needed
mkdir -p "$(pwd)/metal"

# Download Metal shader resources if not present
if [ ! -f "$(pwd)/metal/metal_kernels.metallib" ]; then
    echo "Metal shader library not found. Attempting to download and compile..."
    # Create directory again just in case
    mkdir -p "$(pwd)/metal"

    # Download metal source file from llama.cpp repository
    echo "Downloading ggml-metal.metal..."
    curl -L https://github.com/ggerganov/llama.cpp/raw/master/ggml-metal.metal \
        -o "$(pwd)/metal/ggml-metal.metal"

    # Check if download was successful
    if [ -f "$(pwd)/metal/ggml-metal.metal" ]; then
        # Compile Metal shaders if xcrun is available
        if command -v xcrun &> /dev/null; then
            echo "Compiling Metal shaders using xcrun..."
            if xcrun -sdk macosx metal -c "$(pwd)/metal/ggml-metal.metal" -o "$(pwd)/metal/ggml-metal.air"; then
                if xcrun -sdk macosx metallib "$(pwd)/metal/ggml-metal.air" -o "$(pwd)/metal/metal_kernels.metallib"; then
                    echo "Metal shader library compiled successfully: $(pwd)/metal/metal_kernels.metallib"
                else
                    echo "ERROR: Failed to create Metal library from .air file."
                    # Clean up intermediate file
                    rm -f "$(pwd)/metal/ggml-metal.air"
                fi
            else
                echo "ERROR: Failed to compile Metal source file."
            fi
            # Clean up source file after attempting compilation
            # rm -f "$(pwd)/metal/ggml-metal.metal" # Keep source for potential re-compilation? Optional.
        else
            echo "WARNING: xcrun (Xcode Command Line Tools) not found."
            echo "Cannot compile Metal shaders automatically."
            echo "Please install Xcode Command Line Tools (xcode-select --install) and run this script again,"
            echo "or manually compile ggml-metal.metal into metal_kernels.metallib in the 'metal' directory."
            echo "Metal acceleration may be suboptimal without compiled shaders."
        fi
    else
        echo "ERROR: Failed to download ggml-metal.metal."
    fi
fi

# Run the application
echo "Starting LlamaCag UI..."
python3 main.py "$@"
