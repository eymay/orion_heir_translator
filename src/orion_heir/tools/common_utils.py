"""
Common utilities for the Orion-HEIR translator tools.

This module provides shared functionality used by various command-line tools
and utilities in the translator toolkit.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and normalize a file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file must exist but doesn't
        ValueError: If path is invalid
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    return path


def create_output_directory(dir_path: Path):
    """
    Create output directory if it doesn't exist.
    
    Args:
        dir_path: Directory path to create
    """
    if dir_path and not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output directory: {dir_path}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                      suffix: str = '', length: int = 50, fill: str = '█'):
    """
    Print a progress bar to stdout.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    # Print new line on complete
    if iteration == total:
        print()


class ProgressReporter:
    """Simple progress reporter for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current += increment
        self._print_progress()
    
    def set_progress(self, value: int):
        """Set absolute progress value."""
        self.current = value
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress."""
        print_progress_bar(
            self.current, self.total,
            prefix=self.description,
            suffix=f"({self.current}/{self.total})"
        )
    
    def finish(self, message: str = "Complete"):
        """Finish progress reporting."""
        self.current = self.total
        self._print_progress()
        print(f"\n✅ {message}")


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    optional_deps = []
    
    # Check required dependencies
    try:
        import xdsl
    except ImportError:
        missing_deps.append('xdsl')
    
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import yaml
    except ImportError:
        missing_deps.append('pyyaml')
    
    try:
        import click
    except ImportError:
        missing_deps.append('click')
    
    # Check optional dependencies
    try:
        import orion
    except ImportError:
        optional_deps.append('orion-fhe')
    
    # Report results
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall with: pip install " + " ".join(missing_deps))
        return False
    
    if optional_deps:
        print("⚠️ Missing optional dependencies:")
        for dep in optional_deps:
            print(f"   - {dep}")
        print("These are optional and only needed for specific functionality.")
    
    print("✅ All required dependencies available")
    return True


def get_system_info() -> dict:
    """Get system information for debugging."""
    import platform
    import sys
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'architecture': platform.architecture(),
        'processor': platform.processor(),
    }
    
    # Add package versions
    try:
        import xdsl
        info['xdsl_version'] = getattr(xdsl, '__version__', 'unknown')
    except ImportError:
        info['xdsl_version'] = 'not installed'
    
    try:
        import torch
        info['torch_version'] = torch.__version__
    except ImportError:
        info['torch_version'] = 'not installed'
    
    try:
        import yaml
        info['pyyaml_version'] = getattr(yaml, '__version__', 'unknown')
    except ImportError:
        info['pyyaml_version'] = 'not installed'
    
    return info


def print_system_info():
    """Print system information for debugging."""
    info = get_system_info()
    
    print("System Information")
    print("==================")
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


def safe_import(module_name: str, package: Optional[str] = None):
    """
    Safely import a module with error handling.
    
    Args:
        module_name: Name of module to import
        package: Package name for relative imports
        
    Returns:
        Imported module or None if import fails
    """
    try:
        if package:
            from importlib import import_module
            return import_module(module_name, package)
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"⚠️ Failed to import {module_name}: {e}")
        return None


def ensure_directory_exists(path: Path):
    """Ensure a directory exists, creating it if necessary."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise ValueError(f"Path exists but is not a directory: {path}")


def read_file_safely(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
    """
    Safely read a file with error handling.
    
    Args:
        file_path: Path to file
        encoding: File encoding
        
    Returns:
        File contents or None if read fails
    """
    try:
        return file_path.read_text(encoding=encoding)
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return None


def write_file_safely(file_path: Path, content: str, encoding: str = 'utf-8') -> bool:
    """
    Safely write content to a file with error handling.
    
    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding
        
    Returns:
        True if write succeeded, False otherwise
    """
    try:
        # Ensure parent directory exists
        ensure_directory_exists(file_path.parent)
        
        file_path.write_text(content, encoding=encoding)
        return True
    except Exception as e:
        print(f"❌ Error writing file {file_path}: {e}")
        return False
