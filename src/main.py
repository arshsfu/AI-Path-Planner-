#!/usr/bin/env python3

import os
import sys

if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/sample_maps", exist_ok=True)
    
    print("=" * 70)
    print("AI PATH PLANNER - MILESTONE 2")
    print("Simultaneous Animation of All Algorithms")
    print("=" * 70)
    
    try:
        import run
        run.main()
        
    except KeyboardInterrupt:
        print("\n\nAnimation interrupted by user.")
        sys.exit(0)
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have required packages: pip install matplotlib numpy")
        print("  2. Check if all module files are present in the workspace")
        print("  3. Try running directly: python run.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install matplotlib numpy")
        print("  2. Check if GUI/display is available (required for visualization)")
        print("  3. Try running: python run.py")
        sys.exit(1)

