#!/usr/bin/env python3
"""
Test real detection with cache clearing
"""
import os
import sys
import shutil

# Clear Python cache to ensure we use the updated code
print("🧹 Clearing Python cache...")
cache_dirs = []

# Find all __pycache__ directories
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        cache_path = os.path.join(root, '__pycache__')
        cache_dirs.append(cache_path)

for cache_dir in cache_dirs:
    try:
        shutil.rmtree(cache_dir)
        print(f"  [OK] Removed: {cache_dir}")
    except Exception as e:
        print(f"  [FAIL] Failed to remove {cache_dir}: {e}")

print(f"\n[SUCCESS] Cleared {len(cache_dirs)} cache directories")
print("\n" + "=" * 60)
print("Now restart your detection script to use the updated code!")
print("=" * 60)
