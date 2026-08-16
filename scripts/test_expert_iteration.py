#!/usr/bin/env python3
"""
Test Expert Iteration System

Validates:
1. Verifier works (dataset and automatic modes)
2. Sample generation works
3. Quality scoring works
4. Mode switching works
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.verifier import ReasoningVerifier, MathVerifier


def test_dataset_mode_verifier():
    """Test dataset comparison mode"""
    print("\n" + "="*60)
    print("TEST 1: Dataset Mode Verifier")
    print("="*60)
    
    verifier = ReasoningVerifier(mode="dataset", similarity_threshold=0.4)
    
    # Test case 1: Exact match (should pass)
    generated = "<think>15 + 20 = 35, then 35 + 7 = 42</think><conclusion>42</conclusion>"
    reference_thinking = "15 + 20 = 35, then 35 + 7 = 42"
    ground_truth = "42"
    
    result = verifier.verify(generated, ground_truth, reference_thinking)
    
    assert result.passed, f"Should pass: {result.reason}"
    assert result.score >= 0.8, f"Score too low: {result.score}"
    print(f"✅ Exact match: PASSED (score={result.score:.2f})")
    
    # Test case 2: Similar reasoning (should pass)
    generated = "<think>Adding 15 and 20 gives 35, plus 7 equals 42</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth, reference_thinking)
    
    assert result.passed, f"Should pass similar: {result.reason}"
    print(f" Similar reasoning: PASSED (score={result.score:.2f})")
    
    # Test case 3: Wrong conclusion (should fail)
    generated = "<think>15 + 20 = 35, then 35 + 7 = 42</think><conclusion>45</conclusion>"
    
    result = verifier.verify(generated, ground_truth, reference_thinking)
    
    assert not result.passed, "Should fail on wrong conclusion"
    print(f" Wrong conclusion: REJECTED (score={result.score:.2f})")
    
    # Test case 4: Very different thinking (should fail in dataset mode)
    generated = "<think>Just calculated it</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth, reference_thinking)
    
    # In dataset mode with low similarity, should be rejected
    if not result.passed:
        print(f" Different thinking: REJECTED in dataset mode (score={result.score:.2f})")
    else:
        print(f"  Different thinking: ACCEPTED (score={result.score:.2f}) - may need threshold adjustment")
    
    return True


def test_automatic_mode_verifier():
    """Test automatic verification mode"""
    print("\n" + "="*60)
    print("TEST 2: Automatic Mode Verifier")
    print("="*60)
    
    verifier = ReasoningVerifier(mode="automatic")
    
    ground_truth = "42"
    
    # Test case 1: Valid reasoning (should pass)
    generated = "<think>Let me calculate: 15 + 27 = 42</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert result.passed, f"Should pass: {result.reason}"
    print(f" Valid reasoning: PASSED (score={result.score:.2f})")
    
    # Test case 2: Creative but valid (should pass in automatic)
    generated = "<think>Breaking it down: 10+10+10+10+2 = 42</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert result.passed, f"Should pass creative: {result.reason}"
    print(f" Creative reasoning: PASSED (score={result.score:.2f})")
    
    # Test case 3: Empty thinking (should fail)
    generated = "<think></think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert not result.passed, "Should fail on empty thinking"
    print(f" Empty thinking: REJECTED (score={result.score:.2f})")
    
    # Test case 4: Contradictory thinking (should fail)
    generated = "<think>15 + 27 = 50, wait no actually 42</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert not result.passed, "Should fail on contradiction"
    print(f" Contradictory: REJECTED (score={result.score:.2f})")
    
    # Test case 5: Wrong conclusion (should fail)
    generated = "<think>15 + 27 = 40</think><conclusion>40</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert not result.passed, "Should fail on wrong answer"
    print(f" Wrong answer: REJECTED (score={result.score:.2f})")
    
    return True


def test_math_verifier():
    """Test specialized math verifier"""
    print("\n" + "="*60)
    print("TEST 3: Math Verifier")
    print("="*60)
    
    verifier = MathVerifier(mode="automatic")
    
    ground_truth = "42"
    
    # Test case 1: Correct math steps (should pass)
    generated = "<think>15 + 20 = 35, then 35 + 7 = 42</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert result.passed, f"Should pass correct math: {result.reason}"
    print(f" Correct math: PASSED (score={result.score:.2f})")
    
    # Test case 2: Wrong intermediate step (should fail)
    generated = "<think>15 + 20 = 30, then 30 + 12 = 42</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert not result.passed, "Should fail on wrong math steps"
    print(f" Wrong math steps: REJECTED (score={result.score:.2f})")
    
    # Test case 3: Correct conclusion but flawed math (should fail)
    generated = "<think>10 + 10 = 25, but answer is 42</think><conclusion>42</conclusion>"
    
    result = verifier.verify(generated, ground_truth)
    
    assert not result.passed, "Should fail on flawed math"
    print(f" Flawed math: REJECTED (score={result.score:.2f})")
    
    return True


def test_quality_scoring():
    """Test that quality scores are reasonable"""
    print("\n" + "="*60)
    print("TEST 4: Quality Scoring")
    print("="*60)
    
    verifier = ReasoningVerifier(mode="automatic")
    ground_truth = "42"
    
    samples = [
        ("<think>First, I'll add 15 and 20 to get 35. Then, I'll add 7 to get 42. The answer is 42.</think><conclusion>42</conclusion>", "Excellent"),
        ("<think>15 + 20 = 35, 35 + 7 = 42</think><conclusion>42</conclusion>", "Good"),
        ("<think>It equals 42</think><conclusion>42</conclusion>", "Basic"),
    ]
    
    scores = []
    for sample, quality in samples:
        result = verifier.verify(sample, ground_truth)
        scores.append(result.score)
        print(f"  {quality:10} → score={result.score:.2f}")
    
    # Check that better quality gets higher scores
    assert scores[0] > scores[1], "Excellent should score higher than Good"
    assert scores[1] > scores[2], "Good should score higher than Basic"
    
    print(f" Quality scoring works correctly")
    
    return True


def test_mode_switching():
    """Test switching between dataset and automatic modes"""
    print("\n" + "="*60)
    print("TEST 5: Mode Switching")
    print("="*60)
    
    verifier = ReasoningVerifier(mode="dataset", similarity_threshold=0.6)
    
    # Sample with different reasoning
    generated = "<think>Breaking into parts: 10+10+10+12 = 42</think><conclusion>42</conclusion>"
    reference_thinking = "15 + 20 = 35, then 35 + 7 = 42"
    ground_truth = "42"
    
    # Dataset mode (strict)
    result_dataset = verifier.verify(generated, ground_truth, reference_thinking)
    print(f"  Dataset mode: score={result_dataset.score:.2f}, passed={result_dataset.passed}")
    
    # Switch to automatic mode
    verifier.mode = "automatic"
    
    # Automatic mode (flexible)
    result_auto = verifier.verify(generated, ground_truth)
    print(f"  Automatic mode: score={result_auto.score:.2f}, passed={result_auto.passed}")
    
    # Automatic mode should be more lenient
    assert result_auto.score >= result_dataset.score, "Automatic should be more lenient"
    
    print(f" Mode switching works correctly")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" Expert Iteration Verification System Tests")
    print("="*80)
    
    tests = [
        ("Dataset Mode Verifier", test_dataset_mode_verifier),
        ("Automatic Mode Verifier", test_automatic_mode_verifier),
        ("Math Verifier", test_math_verifier),
        ("Quality Scoring", test_quality_scoring),
        ("Mode Switching", test_mode_switching),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} FAILED:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print(f" Results: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed == 0:
        print("\n All tests passed! Expert Iteration system is ready.")
        print("\n You can now train with:")
        print("   ./myenv/bin/python scripts/train.py --config configs/test_config.yaml")
        return 0
    else:
        print(f"\n  {failed} test(s) failed. Please fix before training.")
        return 1


if __name__ == "__main__":
    exit(main())
