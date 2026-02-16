#!/usr/bin/env python3
"""
Generate HIGH-QUALITY sample data for testing.
Includes reasoning examples, chain-of-thought, math problems, and code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random


# High-quality reasoning examples (manually curated)
REASONING_EXAMPLES = [
    {
        "question": "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
        "reasoning": "To find distance, I need to use the formula: distance = speed × time. The speed is 60 mph and time is 2.5 hours. So distance = 60 × 2.5 = 150 miles.",
        "answer": "The train travels 150 miles."
    },
    {
        "question": "A baker makes 24 cookies and wants to divide them equally among 6 people. How many cookies does each person get?",
        "reasoning": "To divide equally, I need to use division. Total cookies = 24, Number of people = 6. Cookies per person = 24 ÷ 6 = 4.",
        "answer": "Each person gets 4 cookies."
    },
    {
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "reasoning": "This is a trick question that requires careful analysis. If 5 machines make 5 widgets in 5 minutes, then each machine makes 1 widget in 5 minutes. Therefore, 100 machines would each make 1 widget in 5 minutes, producing 100 widgets total.",
        "answer": "It would take 5 minutes."
    },
    {
        "question": "What is 15% of 200?",
        "reasoning": "To find 15% of 200, I convert the percentage to a decimal: 15% = 0.15. Then multiply: 0.15 × 200 = 30.",
        "answer": "15% of 200 is 30."
    },
    {
        "question": "A rectangle has a length of 12 cm and width of 5 cm. What is its area?",
        "reasoning": "The area of a rectangle is calculated using the formula: Area = length × width. Given length = 12 cm and width = 5 cm, so Area = 12 × 5 = 60 square cm.",
        "answer": "The area is 60 square centimeters."
    },
    {
        "question": "If you buy 3 books at $12 each and get a 10% discount, how much do you pay?",
        "reasoning": "First, calculate the total before discount: 3 × $12 = $36. Then calculate the discount: 10% of $36 = 0.10 × 36 = $3.60. Finally, subtract the discount: $36 - $3.60 = $32.40.",
        "answer": "You pay $32.40."
    },
    {
        "question": "What comes next in the sequence: 2, 4, 8, 16, 32, ?",
        "reasoning": "Looking at the pattern, each number is double the previous number. 2 × 2 = 4, 4 × 2 = 8, 8 × 2 = 16, 16 × 2 = 32. So the next number would be 32 × 2 = 64.",
        "answer": "The next number is 64."
    },
    {
        "question": "If a pizza is cut into 8 equal slices and you eat 3 slices, what fraction of the pizza remains?",
        "reasoning": "Total slices = 8. Slices eaten = 3. Slices remaining = 8 - 3 = 5. The fraction remaining is 5/8 of the pizza.",
        "answer": "5/8 of the pizza remains."
    },
]

MATH_PROBLEMS = [
    {
        "problem": "Solve: 2x + 5 = 13",
        "solution": "To solve for x: 2x + 5 = 13. Subtract 5 from both sides: 2x = 8. Divide both sides by 2: x = 4.",
        "answer": "x = 4"
    },
    {
        "problem": "Calculate: (3 + 4) × 2 - 1",
        "solution": "Following order of operations (PEMDAS): First parentheses: 3 + 4 = 7. Then multiplication: 7 × 2 = 14. Finally subtraction: 14 - 1 = 13.",
        "answer": "The result is 13."
    },
    {
        "problem": "What is the square root of 144?",
        "solution": "The square root of 144 is the number that, when multiplied by itself, equals 144. 12 × 12 = 144, so √144 = 12.",
        "answer": "12"
    },
    {
        "problem": "Convert 3/4 to a decimal.",
        "solution": "To convert a fraction to decimal, divide the numerator by the denominator: 3 ÷ 4 = 0.75.",
        "answer": "0.75"
    },
]

CODE_EXAMPLES = [
    {
        "task": "Write a Python function to check if a number is even.",
        "code": "def is_even(n):\n    return n % 2 == 0\n\n# The modulo operator % returns the remainder of division. If a number divided by 2 has remainder 0, it's even.",
    },
    {
        "task": "Write a Python function to find the maximum of two numbers.",
        "code": "def max_of_two(a, b):\n    if a > b:\n        return a\n    else:\n        return b\n\n# Can also use Python's built-in: max(a, b)",
    },
    {
        "task": "Write a Python function to reverse a string.",
        "code": "def reverse_string(s):\n    return s[::-1]\n\n# The slice notation [::-1] reverses the string by stepping backwards.",
    },
]

GENERAL_KNOWLEDGE = [
    "The Earth orbits around the Sun, completing one full revolution approximately every 365.25 days, which we call a year.",
    "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level under normal atmospheric pressure.",
    "The human body has 206 bones in adulthood, though babies are born with about 270 bones that fuse together as they grow.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second, often denoted as 'c' in physics equations.",
]


def generate_reasoning_data(num_examples=100):
    """Generate reasoning examples in chain-of-thought format."""
    data = []
    
    for _ in range(num_examples):
        example = random.choice(REASONING_EXAMPLES)
        
        # Format with special reasoning tokens
        text = f"{example['question']}\n"
        text += f"<think>{example['reasoning']}</think>\n"
        text += f"<conclusion>{example['answer']}</conclusion>"
        
        data.append({"text": text, "type": "reasoning"})
    
    return data


def generate_math_data(num_examples=80):
    """Generate math problem examples."""
    data = []
    
    for _ in range(num_examples):
        problem = random.choice(MATH_PROBLEMS)
        
        text = f"Problem: {problem['problem']}\n"
        text += f"<think>{problem['solution']}</think>\n"
        text += f"<conclusion>{problem['answer']}</conclusion>"
        
        data.append({"text": text, "type": "math"})
    
    return data


def generate_code_data(num_examples=50):
    """Generate code examples."""
    data = []
    
    for _ in range(num_examples):
        code_ex = random.choice(CODE_EXAMPLES)
        
        text = f"Task: {code_ex['task']}\n\n{code_ex['code']}"
        
        data.append({"text": text, "type": "code"})
    
    return data


def generate_general_data(num_examples=70):
    """Generate general knowledge examples."""
    data = []
    
    for _ in range(num_examples):
        text = random.choice(GENERAL_KNOWLEDGE)
        data.append({"text": text, "type": "general"})
    
    return data


def main():
    print("=" * 60)
    print("High-Quality Sample Data Generator")
    print("=" * 60)
    
    # Create directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/eval").mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating high-quality sample data...")
    
    # Generate different types of data
    reasoning_data = generate_reasoning_data(100)
    math_data = generate_math_data(80)
    code_data = generate_code_data(50)
    general_data = generate_general_data(70)
    
    # Combine all data
    all_data = reasoning_data + math_data + code_data + general_data
    random.shuffle(all_data)
    
    # Split into train/eval (90/10 split)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    
    print(f"\nData distribution:")
    print(f"  Reasoning examples: {len(reasoning_data)}")
    print(f"  Math problems: {len(math_data)}")
    print(f"  Code examples: {len(code_data)}")
    print(f"  General knowledge: {len(general_data)}")
    print(f"  Total: {len(all_data)}")
    print(f"\nSplit:")
    print(f"  Training: {len(train_data)} examples")
    print(f"  Evaluation: {len(eval_data)} examples")
    
    # Save raw text for tokenizer training
    print("\nSaving raw text for tokenizer...")
    with open("data/raw/train_text.txt", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(item["text"] + "\n\n")
    
    # Save JSONL for model training
    print("Saving JSONL for training...")
    with open("data/train/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            json.dump(item, f)
            f.write("\n")
    
    with open("data/eval/eval.jsonl", "w", encoding="utf-8") as f:
        for item in eval_data:
            json.dump(item, f)
            f.write("\n")
    
    # Print few examples
    print("\n" + "=" * 60)
    print("Sample Examples:")
    print("=" * 60)
    
    print("\n[Reasoning Example]")
    print(train_data[0]["text"][:200] + "...")
    
    print("\n[Math Example]")
    math_ex = [x for x in train_data if x["type"] == "math"][0]
    print(math_ex["text"][:200] + "...")
    
    print("\n[Code Example]")
    code_ex = [x for x in train_data if x["type"] == "code"][0]
    print(code_ex["text"][:200] + "...")
    
    print("\n" + "=" * 60)
    print("✓ High-quality sample data generated!")
    print("=" * 60)
    print("\nFiles created:")
    print("  data/raw/train_text.txt - Raw text for tokenizer")
    print("  data/train/train.jsonl - Training data")
    print("  data/eval/eval.jsonl - Evaluation data")
    print("\nNext steps:")
    print("  1. Validate loss function: python scripts/validate_loss.py")
    print("  2. Train tokenizer: python scripts/train_tokenizer.py")
    print("  3. Start training: python scripts/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
