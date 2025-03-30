#!/usr/bin/env python3
"""
Configuration management for LlamaCag UI.
Manages application configuration, including loading from .env file.
"""
import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Optional, Any
import dotenv
class ConfigManager:
    """Manages application configuration"""
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration manager"""
        # Default paths
        self.env_file = env_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        self.user_config_dir = os.path.expanduser('~/.llamacag')
        self.user_config_file = os.path.join(self.user_config_dir, 'config.json')
        # Create user config directory if it doesn't exist
        os.makedirs(self.user_config_dir, exist_ok=True)
        # Load the .env file
        self.env_vars = self._load_env_file()
        # Load user config
        self.user_config = self._load_user_config()
        # Merged config
        self.config = {**self.env_vars, **self.user_config}
    def _load_env_file(self) -> Dict[str, Any]:
        """Load environment variables from .env file"""
        # Check if .env file exists
        if not os.path.exists(self.env_file):
            # Try to find .env in the parent directory
            parent_env = os.path.join(os.path.dirname(os.path.dirname(self.env_file)), '.env')
            if os.path.exists(parent_env):
                self.env_file = parent_env
            else:
                # Try to create from example
                example_env = os.path.join(os.path.dirname(self.env_file), '.env.example')
                if os.path.exists(example_env):
                    with open(example_env, 'r') as src, open(self.env_file, 'w') as dst:
                        dst.write(src.read())
                        logging.info(f"Created .env file from example at {self.env_file}")
                else:
                    # Create empty .env file
                    with open(self.env_file, 'w') as f:
                        f.write("# LlamaCag UI Configuration\n")
                        logging.info(f"Created empty .env file at {self.env_file}")
        # Load .env file
        dotenv.load_dotenv(self.env_file)
        # Get all environment variables
        env_vars = {}
        with open(self.env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value.strip('"\'')
        return env_vars
    def _load_user_config(self) -> Dict[str, Any]:
        """Load user configuration from file"""
        # Check if user config file exists
        if not os.path.exists(self.user_config_file):
            # Create empty config
            with open(self.user_config_file, 'w') as f:
                json.dump({}, f, indent=2)
            return {}
        # Load user config
        user_config = {} # Initialize empty dict
        try:
            with open(self.user_config_file, 'r') as f:
                user_config = json.load(f)

            # Add default Metal configuration for Apple Silicon on first load (as per plan)
            if sys.platform == 'darwin' and not user_config.get('METAL_CONFIG_INITIALIZED', False):
                logging.info("Initializing default Metal configuration for macOS user.")
                user_config['METAL_ENABLED'] = True
                user_config['METAL_MEMORY_MB'] = 4096 # Default 4GB
                user_config['METAL_PROFILE'] = 'balanced' # Default profile
                user_config['METAL_CONFIG_INITIALIZED'] = True # Flag to prevent re-initialization

                # Save these defaults immediately back to the user config file
                try:
                    with open(self.user_config_file, 'w') as f_save:
                        json.dump(user_config, f_save, indent=2)
                    logging.info("Saved initial default Metal settings to config.json.")
                except Exception as e_save:
                    logging.error(f"Failed to save initial Metal defaults: {e_save}")
                    # Continue without saving defaults if error occurs

            return user_config # Return the potentially modified config

        except FileNotFoundError:
             # If file doesn't exist yet, create it with defaults (including Metal if applicable)
             logging.info(f"User config file not found at {self.user_config_file}. Creating with defaults.")
             default_config = {}
             if sys.platform == 'darwin':
                 logging.info("Applying default Metal configuration for new macOS user.")
                 default_config['METAL_ENABLED'] = True
                 default_config['METAL_MEMORY_MB'] = 4096
                 default_config['METAL_PROFILE'] = 'balanced'
                 default_config['METAL_CONFIG_INITIALIZED'] = True
             try:
                 with open(self.user_config_file, 'w') as f_create:
                     json.dump(default_config, f_create, indent=2)
                 return default_config
             except Exception as e_create:
                 logging.error(f"Failed to create default user config file: {e_create}")
                 return {} # Return empty if creation fails

        except json.JSONDecodeError as e_json:
             logging.error(f"Error decoding user config JSON: {e_json}. Returning empty config.")
             # Optionally: Backup corrupted file?
             return {}
        except Exception as e:
            logging.error(f"Failed to load user config: {str(e)}")
            return {} # Return empty on other errors

    def save_config(self):
        """Save configuration to files"""
        # Save user config
        try:
            # Only save user config keys, not env vars
            # Extract user config keys from merged config
            user_keys = set(self.user_config.keys())
            save_config = {k: self.config[k] for k in user_keys if k in self.config}
            # Add any new keys that aren't in env_vars
            env_keys = set(self.env_vars.keys())
            new_keys = set(self.config.keys()) - env_keys - user_keys
            for key in new_keys:
                save_config[key] = self.config[key]
            with open(self.user_config_file, 'w') as f:
                json.dump(save_config, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save user config: {str(e)}")
    def get_config(self) -> Dict[str, Any]:
        """Get the merged configuration"""
        return self.config
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)
    def set(self, key: str, value: Any):
        """Set a configuration value"""
        self.config[key] = value
        # If it's an environment variable, update .env file
        if key in self.env_vars:
            self.env_vars[key] = value
            self._save_env_file()
        else:
            # Otherwise, update user config
            self.user_config[key] = value
            self.save_config()
    def _save_env_file(self):
        """Save environment variables to .env file"""
        try:
            # Read current .env file to preserve comments and formatting
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
            # Update values
            new_lines = []
            for line in lines:
                line = line.rstrip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=', 1)[0].strip()
                    if key in self.env_vars:
                        value = self.env_vars[key]
                        line = f"{key}={value}"
                new_lines.append(line)
            # Write back
            with open(self.env_file, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
        except Exception as e:
            logging.error(f"Failed to save .env file: {str(e)}")

    def get_model_specific_config(self, model_id: str) -> Dict[str, Any]:
        """Get model-specific configuration parameters"""
        # Load the base model-specific configs from the main config
        # This allows users to define overrides in config.json or .env
        model_configs = self.get('MODEL_SPECIFIC_CONFIGS', {})

        # Default configuration applicable to all models
        default_config = {
            'eos_detection_method': 'default',  # Options: default, strict, flexible, gemma
            'stop_on_repetition': True,
            'max_empty_tokens': 50,
            'repetition_threshold': 2, # Matches the value set in chat_engine.py
            'additional_stop_tokens': [] # List of extra token IDs to treat as EOS
        }

        # Apply general Gemma defaults if the model ID contains 'gemma'
        if 'gemma' in model_id.lower():
            logging.debug(f"Applying default Gemma config for model: {model_id}")
            default_config['eos_detection_method'] = 'gemma'
            # Common special tokens: 0=PAD, 1=BOS, 2=EOS, 11=special?
            # Let's add 1 (BOS) and 11 (often unused/special) as potential stop tokens for Gemma
            default_config['additional_stop_tokens'] = [1, 11]
            # Gemma might be more prone to repetition? Keep threshold low for now.
            # default_config['repetition_threshold'] = 2

        # Apply specific overrides if defined for this exact model_id
        if model_id in model_configs:
            logging.debug(f"Applying specific config overrides for model: {model_id}")
            model_specific_overrides = model_configs[model_id]
            # Ensure override values have the correct type if necessary
            # For example, ensure lists are lists, bools are bools, ints are ints
            for key, value in model_specific_overrides.items():
                if key in default_config:
                    expected_type = type(default_config[key])
                    try:
                        # Attempt type conversion if types don't match
                        if not isinstance(value, expected_type):
                            if expected_type == bool:
                                converted_value = str(value).lower() in ['true', '1', 'yes']
                            elif expected_type == list:
                                # Handle comma-separated strings for lists
                                if isinstance(value, str):
                                    converted_value = [int(x.strip()) for x in value.split(',') if x.strip()]
                                else:
                                    converted_value = list(value) # Try direct conversion
                            else:
                                converted_value = expected_type(value)
                            default_config[key] = converted_value
                            logging.debug(f"Converted override '{key}': {value} -> {converted_value}")
                        else:
                            default_config[key] = value # Types match, use directly
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Could not apply override for '{key}' due to type mismatch or conversion error: {e}. Using default: {default_config[key]}")
                else:
                    # Allow adding new keys not in the default dict
                    default_config[key] = value

        logging.debug(f"Final model-specific config for '{model_id}': {default_config}")
        return default_config
