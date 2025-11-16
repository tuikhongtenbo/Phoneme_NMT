"""
Kaggle Training Script
Quick start script để train trên Kaggle Notebook
"""

import os
import sys
from pathlib import Path
import subprocess

def setup_environment():
    """Setup environment cho Kaggle."""
    WORK_DIR = Path("/kaggle/working")
    INPUT_DIR = Path("/kaggle/input")
    
    print("=" * 80)
    print("Kaggle Training Setup")
    print("=" * 80)
    
    # Option 1: Clone từ GitHub
    repo_url = os.environ.get("GITHUB_REPO", "https://github.com/tuikhongtenbo/Phoneme_NMT.git")
    repo_dir = WORK_DIR / "Phoneme_NMT"
    
    if not repo_dir.exists():
        print(f"\n📥 Cloning repository from {repo_url}...")
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(repo_dir)],
                check=True,
                capture_output=True
            )
            print("✓ Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Could not clone from GitHub: {e}")
            print("   Trying to use dataset instead...")
            # Option 2: Sử dụng dataset
            dataset_dirs = list(INPUT_DIR.glob("*"))
            if dataset_dirs:
                code_dir = dataset_dirs[0]
                print(f"✓ Using dataset from {code_dir}")
                sys.path.insert(0, str(code_dir))
                os.chdir(code_dir)
                return code_dir
            else:
                raise FileNotFoundError("No code found. Please either:")
                print("   1. Set GITHUB_REPO environment variable")
                print("   2. Upload code as Kaggle dataset")
    else:
        print(f"✓ Repository already exists at {repo_dir}")
        # Pull latest changes
        try:
            subprocess.run(
                ["git", "-C", str(repo_dir), "pull"],
                check=True,
                capture_output=True
            )
        except:
            pass
    
    # Add to path and change directory
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)
    
    print(f"✓ Working directory: {os.getcwd()}")
    return repo_dir


def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    
    packages = [
        "torch", "torchvision", "torchaudio",
        "numpy", "pandas", "tqdm",
        "pydantic", "pyyaml",
    ]
    
    for package in packages:
        try:
            subprocess.run(
                ["pip", "install", "-q", package],
                check=True,
                capture_output=True
            )
        except:
            print(f"⚠️  Could not install {package}, may already be installed")
    
    print("✓ Dependencies installed")


def check_gpu():
    """Check GPU availability."""
    import torch
    print("\n🖥️  GPU Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  No GPU available, training will be slow")


def update_config_for_kaggle():
    """Update config paths for Kaggle environment."""
    print("\n⚙️  Updating config for Kaggle...")
    
    # Paths đã được update trong các file yaml (trỏ đến /kaggle/input/phomt/)
    # Chỉ cần kiểm tra xem dataset có tồn tại không
    kaggle_data_path = Path("/kaggle/input/phomt")
    
    if kaggle_data_path.exists():
        print(f"   ✓ Found dataset at: {kaggle_data_path}")
        return kaggle_data_path
    else:
        print(f"   ⚠️  Dataset not found at {kaggle_data_path}")
        print("   Please ensure dataset 'phomt' is added to Kaggle Notebook")
        return None


def main():
    """Main function."""
    # Setup
    repo_dir = setup_environment()
    install_dependencies()
    check_gpu()
    data_dir = update_config_for_kaggle()
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    # Import và chạy training
    # Sử dụng main.py với config file
    # Parse arguments (có thể set trong Kaggle notebook)
    config_file = os.environ.get("CONFIG_FILE", "configs/lstm_luong.yaml")
    
    print(f"\n📋 Using config: {config_file}")
    print("   (Set CONFIG_FILE environment variable to change)")
    
    # Import main và chạy với config
    from main import main as train_main
    import sys
    
    # Set sys.argv để main.py nhận được config
    original_argv = sys.argv
    sys.argv = ["main.py", "--config", config_file]
    
    try:
        train_main()
    finally:
        sys.argv = original_argv
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\n📁 Results saved to:")
    print(f"   Checkpoints: /kaggle/working/checkpoints/")
    print(f"   Logs: /kaggle/working/logs/")
    print("\n💡 Tip: Download results from Kaggle Notebook output")


if __name__ == "__main__":
    main()

