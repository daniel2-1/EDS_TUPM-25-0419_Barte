"""
=============================================================
  EDS PROJECT — Package Installer
  Run this ONCE before running main.py
  Usage: python install_packages.py
=============================================================
"""

import subprocess
import sys

# ── List of all required packages ────────────────────────────────────────────
PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "scikit-learn",
    "pillow",        # needed for saving .gif animations
]

def install(package: str):
    print(f"  Installing {package}...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", package],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Done")
    else:
        print(f"❌ FAILED\n  Error: {result.stderr.strip()}")

def verify(package: str) -> bool:
    """Check if a package can be imported successfully."""
    import importlib
    # Map pip name → import name
    import_names = {
        "scikit-learn": "sklearn",
        "pillow":        "PIL",
    }
    import_name = import_names.get(package, package)
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 55)
    print("  EDS PROJECT — Installing Required Packages")
    print("=" * 55)
    print(f"  Python: {sys.version}")
    print(f"  Pip:    {sys.executable}\n")

    # ── Install all packages ──────────────────────────────────────────────
    for pkg in PACKAGES:
        install(pkg)

    # ── Verify all imports work ───────────────────────────────────────────
    print("\n── Verifying Imports ──")
    all_ok = True
    for pkg in PACKAGES:
        ok = verify(pkg)
        status = "✅ OK" if ok else "❌ IMPORT FAILED"
        print(f"  {pkg:<20} {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("=" * 55)
        print("  ✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("  You can now run:  python main.py")
        print("=" * 55)
    else:
        print("=" * 55)
        print("  ⚠️  Some packages failed. Try running:")
        print("  pip install -r requirements.txt")
        print("  in your terminal manually.")
        print("=" * 55)

if __name__ == "__main__":
    main()
