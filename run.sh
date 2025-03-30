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
        # First try: Look for metal compiler in standard Xcode locations
        METAL_COMPILER=""
        POTENTIAL_PATHS=(
            # Standard Xcode.app location
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal"
            # Xcode Command Line Tools possible locations
            "/Library/Developer/CommandLineTools/usr/bin/metal"
            "/usr/bin/metal"
        )
        
        for path in "${POTENTIAL_PATHS[@]}"; do
            if [ -x "$path" ]; then
                METAL_COMPILER="$path"
                echo "Found metal compiler at: $METAL_COMPILER"
                break
            fi
        done
        
        # If not found in standard locations, try using xcrun to find it
        if [ -z "$METAL_COMPILER" ]; then
            if command -v xcrun &> /dev/null; then
                # Try to locate metal using xcrun
                METAL_COMPILER=$(xcrun -f metal 2>/dev/null || echo "")
                if [ -n "$METAL_COMPILER" ]; then
                    echo "Found metal compiler via xcrun at: $METAL_COMPILER"
                else
                    echo "WARNING: Could not locate metal compiler using xcrun."
                fi
            fi
        fi
        
        # If we found the metal compiler, use direct path to avoid PATH issues
        if [ -n "$METAL_COMPILER" ]; then
            echo "Compiling Metal shaders using direct compiler path..."
            
            # First compile .metal to .air
            if "$METAL_COMPILER" -c "$(pwd)/metal/ggml-metal.metal" -o "$(pwd)/metal/ggml-metal.air"; then
                # Then convert .air to .metallib using metallib tool
                METALLIB_COMPILER=$(dirname "$METAL_COMPILER")/metallib
                
                if [ -x "$METALLIB_COMPILER" ]; then
                    if "$METALLIB_COMPILER" "$(pwd)/metal/ggml-metal.air" -o "$(pwd)/metal/metal_kernels.metallib"; then
                        echo "Metal shader library compiled successfully: $(pwd)/metal/metal_kernels.metallib"
                    else
                        echo "ERROR: Failed to create Metal library from .air file."
                        rm -f "$(pwd)/metal/ggml-metal.air"
                    fi
                else
                    echo "ERROR: metallib compiler not found at expected location: $METALLIB_COMPILER"
                    rm -f "$(pwd)/metal/ggml-metal.air"
                fi
            else
                echo "ERROR: Failed to compile Metal source file."
            fi
        else
            # Fallback: Try to download pre-compiled metallib directly
            echo "Metal compiler not found. Attempting to download pre-compiled shader library..."
            curl -L https://github.com/ggerganov/llama.cpp/raw/master/metal_kernels.metallib \
                -o "$(pwd)/metal/metal_kernels.metallib"
                
            if [ -f "$(pwd)/metal/metal_kernels.metallib" ]; then
                echo "Downloaded pre-compiled Metal shader library successfully."
                echo "This may not be optimized for your specific hardware, but should work."
            else
                echo "ERROR: Failed to download pre-compiled Metal shader library."
                echo "Metal acceleration will be unavailable. Using CPU only mode."
                echo "To enable Metal acceleration later, you can try:"
                echo "1. Install full Xcode from App Store (not just Command Line Tools)"
                echo "2. Run 'sudo xcode-select --switch /Applications/Xcode.app' in Terminal"
                echo "3. Rerun this application"
            fi
        fi
    else
        echo "ERROR: Failed to download ggml-metal.metal."
    fi
fi

# If Metal shader library still doesn't exist, disable Metal acceleration
if [ ! -f "$(pwd)/metal/metal_kernels.metallib" ]; then
    echo "WARNING: Metal shader library not available, disabling GPU acceleration"
    export GGML_METAL_FULL_BACKEND=0
    export GGML_METAL_PATH_RESOURCES=""
    # Set GPU layers to 0 in environment to ensure CPU-only mode
    export GGML_NGPU=0
    export GGML_GPU_LAYERS=0
fi

# Run the application
echo "Starting LlamaCag UI..."
python3 main.py "$@"
