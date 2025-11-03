"""
Test to see what labels the agent tools actually see when they load models.
This will help identify if there's a cache/blob storage mismatch.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test what the agent tools see
from Agents.ImageAgents.tools.dima_vit_detect_tool import load_dima_model
from Agents.ImageAgents.tools.prithiv_v1_detect_tool import load_prithiv_model
from Agents.ImageAgents.tools.organika_sdxl_detect_tool import load_organika_model

print("\n" + "="*80)
print("TESTING AGENT TOOL MODEL LABELS")
print("="*80)

print("\n--- Prithiv V1 (WORKING) ---")
try:
    model, processor, device = load_prithiv_model()
    print(f"✓ Model loaded")
    print(f"Labels: {model.config.id2label}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n--- DimaVit (BROKEN) ---")
try:
    model, processor, device = load_dima_model()
    print(f"✓ Model loaded")
    print(f"Labels: {model.config.id2label}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n--- Organika SDXL (BROKEN) ---")
try:
    model, processor, device = load_organika_model()
    print(f"✓ Model loaded")
    print(f"Labels: {model.config.id2label}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
