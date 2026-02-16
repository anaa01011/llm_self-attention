#!/usr/bin/env python3
"""
Reasoning Verification System

Two modes:
1. Dataset-based: Compare against reference thinking (early training)
2. Automatic: Verify conclusion + quality checks (after model learns)
"""

import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of verification"""
    score: float  # 0.0 to 1.0
    passed: bool  # True if should train on this sample
    reason: str   # Why this score was given
    
    
class ReasoningVerifier:
    """
    Verifies model-generated reasoning against ground truth.
    
    Two modes:
    - dataset: Compare thinking similarity to reference (early training)
    - automatic: Verify conclusion + quality only (later training)
    """
    
    def __init__(self, mode: str = "dataset", similarity_threshold: float = 0.4):
        """
        Args:
            mode: "dataset" or "automatic"
            similarity_threshold: Minimum similarity to accept (dataset mode)
        """
        assert mode in ["dataset", "automatic"], f"Invalid mode: {mode}"
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        
    def verify(
        self,
        generated_output: str,
        ground_truth_conclusion: str,
        reference_thinking: Optional[str] = None,
        question: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a generated sample.
        
        Args:
            generated_output: Full model output with <think> and <conclusion>
            ground_truth_conclusion: Expected answer
            reference_thinking: Reference thinking from dataset (optional)
            question: Original question (optional, for context)
            
        Returns:
            VerificationResult with score and decision
        """
        # Extract thinking and conclusion
        thinking = self._extract_tag(generated_output, "think")
        conclusion = self._extract_tag(generated_output, "conclusion")
        
        if thinking is None or conclusion is None:
            return VerificationResult(
                score=0.0,
                passed=False,
                reason="Missing <think> or <conclusion> tags"
            )
        
        # Mode selection
        if self.mode == "dataset":
            return self._verify_dataset_mode(
                thinking, conclusion, ground_truth_conclusion, reference_thinking
            )
        else:
            return self._verify_automatic_mode(
                thinking, conclusion, ground_truth_conclusion, question
            )
            
    def _verify_dataset_mode(
        self,
        thinking: str,
        conclusion: str,
        ground_truth_conclusion: str,
        reference_thinking: Optional[str],
    ) -> VerificationResult:
        """
        Dataset comparison mode (Phase 1: Early training)
        
        Checks:
        1. Conclusion must match exactly
        2. Thinking should be similar to reference
        """
        # Check conclusion
        conclusion_clean = conclusion.strip().lower()
        ground_truth_clean = ground_truth_conclusion.strip().lower()
        
        if conclusion_clean != ground_truth_clean:
            return VerificationResult(
                score=0.0,
                passed=False,
                reason=f"Wrong conclusion: '{conclusion}' != '{ground_truth_conclusion}'"
            )
        
        # If no reference, fall back to basic checks
        if reference_thinking is None:
            return self._verify_automatic_mode(thinking, conclusion, ground_truth_conclusion, None)
        
        # Compare thinking to reference
        similarity = self._compute_similarity(thinking, reference_thinking)
        
        if similarity >= self.similarity_threshold:
            # High similarity - good learning
            score = 0.6 + (similarity - self.similarity_threshold) / (1.0 - self.similarity_threshold) * 0.4
            return VerificationResult(
                score=score,
                passed=True,
                reason=f"Good match (similarity: {similarity:.2f})"
            )
        else:
            # Too different - might be hallucinating
            return VerificationResult(
                score=similarity,
                passed=False,
                reason=f"Thinking differs from reference (similarity: {similarity:.2f})"
            )
            
    def _verify_automatic_mode(
        self,
        thinking: str,
        conclusion: str,
        ground_truth_conclusion: str,
        question: Optional[str],
    ) -> VerificationResult:
        """
        Automatic verification mode (Phase 2: After model learns)
        
        Checks:
        1. Conclusion must match exactly
        2. Thinking must be valid (not empty, no contradictions)
        3. Thinking should logically support conclusion
        """
        # Check conclusion
        conclusion_clean = conclusion.strip().lower()
        ground_truth_clean = ground_truth_conclusion.strip().lower()
        
        if conclusion_clean != ground_truth_clean:
            return VerificationResult(
                score=0.0,
                passed=False,
                reason=f"Wrong conclusion: '{conclusion}' != '{ground_truth_conclusion}'"
            )
        
        # Basic quality checks
        if len(thinking.strip()) < 5:
            return VerificationResult(
                score=0.0,
                passed=False,
                reason="Thinking is too short or empty"
            )
        
        # Check for contradictions
        if self._has_contradiction(thinking):
            return VerificationResult(
                score=0.0,
                passed=False,
                reason="Thinking contains contradictions"
            )
        
        # Check if conclusion appears in or follows from thinking
        if not self._conclusion_supported(thinking, conclusion):
            return VerificationResult(
                score=0.0,
                passed=False,
                reason="Conclusion not supported by thinking"
            )
        
        # Score by quality
        score = 0.6  # Base score for correct answer + valid reasoning
        
        # Bonus for quality indicators
        if self._has_step_by_step(thinking):
            score += 0.2
        if self._shows_work(thinking):
            score += 0.1
        if self._is_clear(thinking):
            score += 0.1
        
        return VerificationResult(
            score=min(score, 1.0),
            passed=True,
            reason=f"Valid reasoning (quality score: {score:.2f})"
        )
    
    # ============ Helper Functions ============
    
    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        """Extract content between XML-style tags"""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple word-overlap similarity.
        
        More sophisticated: Could use embeddings, but this works for starter.
        """
        # Tokenize
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _has_contradiction(self, thinking: str) -> bool:
        """Check for obvious contradictions in thinking"""
        # Look for negation patterns
        contradiction_patterns = [
            r"(no|not|wrong).{0,20}(wait|actually|correction)",
            r"equals \d+.{0,20}no.{0,20}equals \d+",
            r"is \w+.{0,20}actually.{0,20}is \w+",
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, thinking.lower()):
                return True
        
        return False
    
    def _conclusion_supported(self, thinking: str, conclusion: str) -> bool:
        """Check if conclusion appears in or follows from thinking"""
        thinking_lower = thinking.lower()
        conclusion_lower = conclusion.lower()
        
        # Conclusion mentioned directly
        if conclusion_lower in thinking_lower:
            return True
        
        # For numbers, check if it appears anywhere
        if conclusion.strip().isdigit():
            if conclusion.strip() in re.findall(r'\d+', thinking):
                return True
        
        # If thinking is very short, be lenient
        if len(thinking.strip()) < 30:
            return True
        
        return False
    
    def _has_step_by_step(self, thinking: str) -> bool:
        """Check if thinking shows step-by-step reasoning"""
        step_indicators = [
            r"first|second|third|then|next|finally",
            r"step \d+",
            r"\d+[.)] ",  # Numbered lists
            r"let me|let's|now",
        ]
        
        matches = sum(1 for pattern in step_indicators if re.search(pattern, thinking.lower()))
        return matches >= 2
    
    def _shows_work(self, thinking: str) -> bool:
        """Check if thinking shows intermediate calculations/work"""
        # Look for mathematical expressions or intermediate results
        patterns = [
            r"\d+\s*[+\-*/]\s*\d+\s*=\s*\d+",  # "15 + 20 = 35"
            r"equals \d+",
            r"= \d+",
            r"\d+ is \d+",
            r"to get \d+",
            r"gives \d+",
            r"add \d+",
        ]
        
        matches = 0
        for pattern in patterns:
            if re.search(pattern, thinking):
                matches += 1
        
        # Need at least 2 intermediate results to count as "showing work"
        return matches >= 2
    
    def _is_clear(self, thinking: str) -> bool:
        """Check if thinking is clear and well-structured"""
        # Simple heuristics
        has_proper_capitalization = thinking[0].isupper() if thinking else False
        has_punctuation = any(c in thinking for c in '.!?')
        reasonable_length = 20 <= len(thinking) <= 500
        
        return has_proper_capitalization and has_punctuation and reasonable_length


class MathVerifier(ReasoningVerifier):
    """
    Specialized verifier for mathematical reasoning.
    
    Validates intermediate calculations in thinking.
    """
    
    def _verify_automatic_mode(
        self,
        thinking: str,
        conclusion: str,
        ground_truth_conclusion: str,
        question: Optional[str],
    ) -> VerificationResult:
        """Enhanced verification for math problems"""
        # First do standard checks
        result = super()._verify_automatic_mode(thinking, conclusion, ground_truth_conclusion, question)
        
        if not result.passed:
            return result
        
        # Additional math-specific checks
        if not self._verify_math_steps(thinking):
            return VerificationResult(
                score=0.0,
                passed=False,
                reason="Invalid mathematical steps in thinking"
            )
        
        return result
    
    def _verify_math_steps(self, thinking: str) -> bool:
        """Verify that mathematical expressions in thinking are correct"""
        # Extract expressions like "15 + 20 = 35"
        expressions = re.findall(r'(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)', thinking)
        
        for expr in expressions:
            left, op, right, result = expr
            left, right, result = int(left), int(right), int(result)
            
            # Evaluate
            if op == '+':
                expected = left + right
            elif op == '-':
                expected = left - right
            elif op == '*':
                expected = left * right
            elif op == '/':
                expected = left // right if right != 0 else None
            else:
                continue
            
            # Verify
            if expected != result:
                return False  # Wrong calculation
        
        return True  # All calculations correct or no expressions found
