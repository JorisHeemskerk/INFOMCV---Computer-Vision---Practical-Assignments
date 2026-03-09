import importlib.util
import sys

def check_cuda():
    print("=" * 50)
    print("       CUDA & GPU Availability Checker")
    print("=" * 50)

    # --- PyTorch ---
    if importlib.util.find_spec("torch") is not None:
        import torch
        print(f"\n✅ PyTorch version:     {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available:      {'✅ Yes' if cuda_available else '❌ No'}")
        if cuda_available:
            print(f"   CUDA version:        {torch.version.cuda}")
            print(f"   GPU count:           {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"   GPU {i}:              {name} ({mem:.1f} GB)")
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"   Apple MPS (M-chip):  {'✅ Yes' if mps_available else '❌ No'}")
    else:
        print("\n⚠️  PyTorch is not installed. Run: pip install torch")

    # --- CUDA Toolkit (via nvidia-smi) ---
    print("\n--- nvidia-smi output ---")
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                print(f"   GPU Name:            {parts[0]}")
                print(f"   Driver version:      {parts[1]}")
                print(f"   Total memory:        {parts[2]}")
        else:
            print("   ❌ nvidia-smi not found or no NVIDIA GPU detected.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("   ❌ nvidia-smi not available on this system.")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_cuda()