#!/usr/bin/env python3
"""
llama.cpp management for LlamaCag UI
Handles installation, updates, and version checking for llama.cpp.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
import shutil
import time
from typing import Optional, Tuple, List
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox


class LlamaManager(QObject):
    """Manages llama.cpp installation and updates"""

    # Signals
    installation_progress = pyqtSignal(int, str)  # progress percentage, message
    installation_complete = pyqtSignal(bool, str)  # success, message

    def __init__(self, config):
        """Initialize llama manager"""
        super().__init__()
        self.config = config
        self.llamacpp_path = Path(os.path.expanduser(config.get('LLAMACPP_PATH', '~/Documents/llama.cpp')))
        logging.info(f"Initialized LlamaManager with path: {self.llamacpp_path}")

    def is_installed(self) -> bool:
        """Check if llama.cpp is installed with improved detection"""
        # Check if directory exists
        if not self.llamacpp_path.exists():
            logging.warning(f"llama.cpp directory not found at {self.llamacpp_path}")
            return False
            
        # Check for various possible executable locations
        possible_executables = [
            self.llamacpp_path / 'build' / 'bin' / 'main',  # Standard location
            self.llamacpp_path / 'build' / 'main',          # Alternate location
            self.llamacpp_path / 'main'                     # Simple build location
        ]
        
        # Check for macOS specific locations
        if sys.platform == 'darwin':
            possible_executables.extend([
                self.llamacpp_path / 'build' / 'bin' / 'llama-cli',
                self.llamacpp_path / 'build' / 'llama-cli'
            ])
        
        # Check if any CMake files exist (indication of successful build)
        cmake_files = [
            self.llamacpp_path / 'build' / 'CMakeCache.txt',
            self.llamacpp_path / 'build' / 'cmake_install.cmake'
        ]
        
        # Look for either executables or cmake files as evidence of installation
        for exe in possible_executables:
            if exe.exists():
                logging.info(f"Found llama.cpp executable at {exe}")
                return True
                
        for cmake_file in cmake_files:
            if cmake_file.exists():
                logging.info(f"Found llama.cpp build file at {cmake_file}")
                return True
        
        # As a last resort, check if the source code exists and has been built at all
        if (self.llamacpp_path / 'CMakeLists.txt').exists() and (self.llamacpp_path / 'build').exists():
            logging.info("Found llama.cpp source and build directory")
            return True
            
        logging.warning("No evidence of llama.cpp installation found")
        return False

    def get_version(self) -> str:
        """Get the installed version of llama.cpp"""
        if not self.is_installed():
            return "Not installed"
        try:
            # Try to get version from git
            result = subprocess.run(
                f"cd {self.llamacpp_path} && git describe --tags",
                shell=True, check=True, capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            # Fall back to "unknown"
            return "Unknown"

    def is_update_available(self) -> bool:
        """Check if an update is available for llama.cpp"""
        if not self.is_installed():
            return False
        try:
            # Fetch latest updates
            subprocess.run(
                f"cd {self.llamacpp_path} && git fetch",
                shell=True, check=True, capture_output=True
            )
            # Check if local is behind remote
            result = subprocess.run(
                f"cd {self.llamacpp_path} && git status -uno",
                shell=True, check=True, capture_output=True, text=True
            )
            return "Your branch is behind" in result.stdout
        except Exception as e:
            logging.error(f"Error checking for updates: {str(e)}")
            return False

    def _check_dependencies(self) -> List[str]:
        """Check if required dependencies are installed"""
        dependencies = {
            'git': 'git --version',
            'cmake': 'cmake --version',
            'make': 'make --version'
        }
        
        missing = []
        for dep, cmd in dependencies.items():
            try:
                result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    missing.append(dep)
            except Exception:
                missing.append(dep)
                
        return missing

    def _install_homebrew(self) -> bool:
        """Install Homebrew if not already installed"""
        try:
            # Check if brew is already installed
            result = subprocess.run(
                "which brew", 
                shell=True, 
                check=False, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                return True  # Already installed
                
            # Install Homebrew
            self.installation_progress.emit(10, "Installing Homebrew (this may take a while)...")
            
            homebrew_install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            result = subprocess.run(
                homebrew_install_cmd,
                shell=True,
                check=False,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logging.error(f"Failed to install Homebrew: {result.stderr}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error installing Homebrew: {str(e)}")
            return False

    def _install_dependencies(self, missing_deps: List[str]) -> bool:
        """Install missing dependencies using Homebrew"""
        try:
            # First make sure Homebrew is installed
            if not self._install_homebrew():
                return False
                
            # Install each missing dependency
            for dep in missing_deps:
                self.installation_progress.emit(15, f"Installing {dep} (this may take a while)...")
                
                cmd = f"brew install {dep}"
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logging.error(f"Failed to install {dep}: {result.stderr}")
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Error installing dependencies: {str(e)}")
            return False

    def install(self) -> bool:
        """Install llama.cpp"""
        # Check dependencies first
        missing_deps = self._check_dependencies()
        if missing_deps:
            self.installation_progress.emit(5, f"Checking dependencies... Missing: {', '.join(missing_deps)}")
            
            # Try to install missing dependencies
            if not self._install_dependencies(missing_deps):
                error_msg = f"Failed to install required dependencies: {', '.join(missing_deps)}.\n\nPlease install them manually with:\nbrew install {' '.join(missing_deps)}"
                self.installation_complete.emit(False, error_msg)
                return False
                
            # Re-check dependencies after install attempt
            missing_deps = self._check_dependencies()
            if missing_deps:
                error_msg = f"Still missing dependencies after installation attempt: {', '.join(missing_deps)}.\n\nPlease install them manually with:\nbrew install {' '.join(missing_deps)}"
                self.installation_complete.emit(False, error_msg)
                return False
        
        # Start installation in a separate thread
        import threading
        threading.Thread(
            target=self._install_thread,
            daemon=True
        ).start()
        return True

    def _install_thread(self):
        """Thread function for llama.cpp installation"""
        try:
            # Create directory if it doesn't exist
            self.llamacpp_path = Path(os.path.expanduser(self.llamacpp_path))
            if not self.llamacpp_path.exists():
                os.makedirs(self.llamacpp_path, parents=True)

            # Signal progress
            self.installation_progress.emit(20, "Creating directories...")

            # Clone repository
            if not (self.llamacpp_path / '.git').exists():
                self.installation_progress.emit(30, "Cloning llama.cpp repository...")

                # Use a more reliable git clone command
                cmd = f"git clone https://github.com/ggerganov/llama.cpp.git {self.llamacpp_path}"
                process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

                if process.returncode != 0:
                    raise Exception(f"Git clone failed: {process.stderr}")
            else:
                self.installation_progress.emit(30, "Updating existing repository...")

                # Use a more reliable git pull command
                cmd = f"cd {self.llamacpp_path} && git pull"
                process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

                if process.returncode != 0:
                    raise Exception(f"Git pull failed: {process.stderr}")

            # Create build directory
            build_path = self.llamacpp_path / 'build'
            if not build_path.exists():
                build_path.mkdir(parents=True)

            # Signal progress
            self.installation_progress.emit(50, "Configuring build...")

            # Configure build with better error handling
            cmd = f"cd {self.llamacpp_path} && cmake -B build"
            process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

            if process.returncode != 0:
                raise Exception(f"CMake configuration failed: {process.stderr}")

            # Signal progress
            self.installation_progress.emit(70, "Building llama.cpp (this may take a while)...")

            # Build with better error handling
            cpu_count = os.cpu_count() or 4
            cmd = f"cd {self.llamacpp_path} && cmake --build build -j {cpu_count}"
            process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

            if process.returncode != 0:
                raise Exception(f"Build failed: {process.stderr}")

            # Create models directory
            models_path = self.llamacpp_path / 'models'
            if not models_path.exists():
                models_path.mkdir(parents=True)

            # Signal completion
            self.installation_progress.emit(100, "Installation complete!")
            self.installation_complete.emit(True, "llama.cpp installed successfully!")

        except Exception as e:
            logging.error(f"Installation failed: {str(e)}")
            self.installation_complete.emit(False, f"Installation failed: {str(e)}")

    def update_config(self, config):
        """Update configuration"""
        self.config = config
        self.llamacpp_path = Path(os.path.expanduser(config.get('LLAMACPP_PATH', '~/Documents/llama.cpp')))

    def detect_metal_capabilities(self):
        """Detect Metal capabilities and return recommended settings"""
        # Import necessary modules locally within the method if not already imported globally
        import sys
        import subprocess
        import logging # Assuming logging is already configured
        import os # Added to check for files

        try:
            # Check if on macOS
            if sys.platform != 'darwin':
                return {"supported": False, "reason": "Not running on macOS"}

            # First, check if Metal shader library exists (this is critical)
            metal_lib_path = os.path.join(os.getcwd(), "metal", "metal_kernels.metallib")
            if not os.path.exists(metal_lib_path):
                return {
                    "supported": False, 
                    "reason": f"Metal shader library not found at: {metal_lib_path}. "
                             f"GPU acceleration will be unavailable. Please rerun the application or "
                             f"manually compile Metal shaders."
                }

            # Check for Apple Silicon
            is_apple_silicon = False
            try:
                # Use platform module for more robust check
                import platform
                is_apple_silicon = platform.machine() == 'arm64'
            except Exception as e_plat:
                 logging.warning(f"Could not determine platform machine type: {e_plat}. Falling back to uname.")
                 try:
                     uname_output = subprocess.check_output(['uname', '-m'], text=True, stderr=subprocess.PIPE).strip()
                     is_apple_silicon = uname_output == 'arm64'
                 except (subprocess.CalledProcessError, FileNotFoundError) as e_uname:
                     logging.error(f"Failed to run uname: {e_uname}")
                     # Assume not Apple Silicon if check fails
                     pass

            if not is_apple_silicon:
                return {"supported": False, "reason": "Not running on Apple Silicon (arm64)"}

            # Now check if Metal environment variables are set correctly
            metal_env_vars = {
                "GGML_METAL_PATH_RESOURCES": os.environ.get("GGML_METAL_PATH_RESOURCES", ""),
                "GGML_METAL_FULL_BACKEND": os.environ.get("GGML_METAL_FULL_BACKEND", "")
            }
            
            # If Metal path isn't set or points to a non-existent directory, it's a problem
            if not metal_env_vars["GGML_METAL_PATH_RESOURCES"] or not os.path.exists(metal_env_vars["GGML_METAL_PATH_RESOURCES"]):
                logging.warning(f"Metal resources path not correctly set: {metal_env_vars['GGML_METAL_PATH_RESOURCES']}")
            
            # Get GPU info using system_profiler
            gpu_model = "Unknown"
            gpu_cores = 0
            metal_supported = False
            metal_feature_set = "Unknown"
            try:
                # Use text=True for automatic decoding
                system_profiler_output = subprocess.check_output(
                    ['system_profiler', 'SPDisplaysDataType'],
                    text=True, stderr=subprocess.PIPE
                )

                # Parse the output
                current_gpu_cores = 0 # Track cores for the current GPU block
                for line in system_profiler_output.splitlines():
                    line_stripped = line.strip()
                    if 'Chipset Model:' in line_stripped:
                        gpu_model = line_stripped.split(':', 1)[1].strip()
                        current_gpu_cores = 0 # Reset cores for new GPU
                    elif 'Total Number of Cores:' in line_stripped: # More reliable than 'Cores:'
                        try:
                            current_gpu_cores = int(line_stripped.split(':')[1].strip())
                        except ValueError:
                            logging.warning(f"Could not parse GPU cores from line: {line_stripped}")
                            current_gpu_cores = 0
                    elif 'Metal:' in line_stripped: # Check for Metal support line
                        # Example: Metal: Supported, feature set macOS_GPUFamily2_v1
                        if 'Supported' in line_stripped:
                             metal_supported = True
                             # Try to extract feature set
                             if 'feature set' in line_stripped:
                                 try:
                                     metal_feature_set = line_stripped.split('feature set', 1)[1].strip()
                                 except IndexError:
                                     pass # Keep default "Unknown" if parsing fails
                             # If Metal is supported for this GPU, assign its cores
                             if current_gpu_cores > 0:
                                 gpu_cores = current_gpu_cores
                                 # Break here assuming we found the primary Metal GPU?
                                 # Or continue to find the most capable one if multiple?
                                 # Let's assume the first Metal-supported GPU with cores is the target.
                                 break

            except (subprocess.CalledProcessError, FileNotFoundError) as e_sp:
                logging.error(f"Failed to run system_profiler SPDisplaysDataType: {e_sp}")
                # Proceed with defaults if system_profiler fails

            if not metal_supported:
                 return {"supported": False, "reason": "Metal support not detected via system_profiler"}

            # Verify Metal is actually usable by doing a simple test
            try:
                # Try to load a small Metal test program - this would detect more subtle issues
                test_cmd = """
                echo '#include <metal_stdlib>
                using namespace metal;
                kernel void test_kernel(device float* output, uint index [[thread_position_in_grid]]) {
                    output[index] = 1.0f;
                }' > /tmp/test.metal && xcrun -sdk macosx metal -c /tmp/test.metal -o /tmp/test.air && xcrun -sdk macosx metallib /tmp/test.air -o /tmp/test.metallib
                """
                subprocess.run(test_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # If we got here, Metal compilation works
                logging.info("Metal test compilation succeeded, confirming Metal is usable")
            except subprocess.CalledProcessError:
                logging.warning("Metal test compilation failed, but continuing with detected capabilities")
                # Don't fail completely, as our pre-compiled metallib might still work

            # Determine recommended gpu_layers based on model size and detected cores
            def get_recommended_layers(model_id, detected_gpu_cores):
                # Default conservative value if cores not detected
                cores_to_use = detected_gpu_cores if detected_gpu_cores > 0 else 8

                # Simple logic based on model name identifier
                if '4b' in model_id.lower():
                    # More layers for smaller models, up to available cores
                    return min(cores_to_use, 16) # Cap at 16 as a reasonable upper limit
                elif '7b' in model_id.lower() or '8b' in model_id.lower():
                    # Medium layers for medium models
                    return min(cores_to_use, 12) # Cap at 12
                elif '13b' in model_id.lower() or '15b' in model_id.lower():
                     # Fewer layers for larger models
                     return min(cores_to_use, 10) # Cap at 10
                else:
                    # Conservative default for unknown/larger sizes
                    return min(cores_to_use, 8) # Cap at 8

            return {
                "supported": True,
                "gpu_model": gpu_model,
                "gpu_cores": gpu_cores,
                "feature_set": metal_feature_set,
                "recommended_formats": ["Q4_1", "Q5_K_M", "IQ4_NL"], # General recommendations
                "get_recommended_layers": get_recommended_layers, # Return the function itself
                "metal_lib_path": metal_lib_path, # Return the path to metallib so UI can check it
                "env_configured": bool(metal_env_vars["GGML_METAL_PATH_RESOURCES"] and 
                                      metal_env_vars["GGML_METAL_FULL_BACKEND"] == "1")
            }
        except Exception as e:
            logging.error(f"Error detecting Metal capabilities: {e}", exc_info=True) # Log traceback
            return {"supported": False, "reason": f"Unexpected error: {e}"}
