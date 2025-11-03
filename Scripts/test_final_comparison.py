"""
Final test: Full agent workflow vs imagedetectionlib ensemble
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise

test_image = r"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_13.jpg"

print("\n" + "="*80)
print("FINAL COMPARISON: AGENT WORKFLOW VS IMAGEDETECTIONLIB ENSEMBLE")
print("="*80)
print(f"\nTest image: {os.path.basename(test_image)}")
print(f"Expected: FAKE")

# Test 1: Agent workflow
print("\n" + "-"*80)
print("AGENT WORKFLOW")
print("-"*80)

async def test_agent_workflow():
    from Agents.ImageAgents.image_workflow import run_image_detection_workflow
    result = await run_image_detection_workflow(
        image_path=test_image,
        threshold=0.5,
        voting_method='weighted'
    )
    return result

agent_result = asyncio.run(test_agent_workflow())

print(f"\nFinal Prediction: {agent_result.prediction}")
print(f"Confidence: {agent_result.confidence:.4f}")
print(f"Avg Fake: {agent_result.averageFake:.4f}, Avg Real: {agent_result.averageReal:.4f}")
print(f"Voting: {agent_result.voting.fake} FAKE, {agent_result.voting.real} REAL")
print(f"\nIndividual models:")
for model in agent_result.individualModelResults:
    print(f"  {model.model}: {model.prediction} (fake={model.fakeScore:.4f}, real={model.realScore:.4f})")

# Test 2: imagedetectionlib ensemble
print("\n" + "-"*80)
print("IMAGEDETECTIONLIB ENSEMBLE")
print("-"*80)

from imagedetectionlib.detect_deepfake_image import EnsembleDetector

ensemble = EnsembleDetector(
    model_keys=['nonescape-full', 'seresnet50', 'prithiv-v1', 'dima-vit', 'organika-sdxl'],
    debug=False
)
lib_result = ensemble.detect(test_image, threshold=0.5, voting_method='weighted')

print(f"\nFinal Prediction: {lib_result.prediction}")
print(f"Confidence: {lib_result.confidence:.4f}")
print(f"Avg Fake: {lib_result.averageFake:.4f}, Avg Real: {lib_result.averageReal:.4f}")
print(f"Voting: {lib_result.voting.fake} FAKE, {lib_result.voting.real} REAL")
print(f"\nIndividual models:")
for model in lib_result.individualModelResults:
    print(f"  {model.model}: {model.prediction} (fake={model.fakeScore:.4f}, real={model.realScore:.4f})")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

match = agent_result.prediction == lib_result.prediction == "FAKE"
print(f"\n✅ Both predict FAKE: {match}")
print(f"Agent voting: {agent_result.voting.fake} FAKE vs {agent_result.voting.real} REAL")
print(f"Lib voting: {lib_result.voting.fake} FAKE vs {lib_result.voting.real} REAL")

if agent_result.voting.fake == lib_result.voting.fake:
    print(f"✅ Voting matches!")
else:
    print(f"⚠️ Voting differs!")
    
print("\n" + "="*80)
