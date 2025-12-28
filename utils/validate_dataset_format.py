#!/usr/bin/env python3
"""
Validation script to verify VideoLLaMA3 dataset format.
Checks if qved_train.json, qved_val.json, qved_test.json are properly formatted.
"""

import json
import sys
from pathlib import Path

def validate_sample(sample, split_name, index):
    """Validate a single data sample."""
    errors = []

    # Check required keys
    if 'video' not in sample:
        errors.append(f"Missing 'video' key in {split_name}[{index}]")
    elif not isinstance(sample['video'], list):
        errors.append(f"'video' must be a list in {split_name}[{index}]")
    elif len(sample['video']) == 0:
        errors.append(f"'video' list is empty in {split_name}[{index}]")

    if 'conversations' not in sample:
        errors.append(f"Missing 'conversations' key in {split_name}[{index}]")
    elif not isinstance(sample['conversations'], list):
        errors.append(f"'conversations' must be a list in {split_name}[{index}]")
    elif len(sample['conversations']) < 2:
        errors.append(f"'conversations' must have at least 2 entries in {split_name}[{index}]")
    else:
        # Validate conversation format
        for i, conv in enumerate(sample['conversations']):
            if 'from' not in conv:
                errors.append(f"Missing 'from' key in {split_name}[{index}] conversation {i}")
            if 'value' not in conv:
                errors.append(f"Missing 'value' key in {split_name}[{index}] conversation {i}")

        # Check for <video> tag in first human message
        if len(sample['conversations']) > 0:
            first_msg = sample['conversations'][0]
            if first_msg.get('from') == 'human':
                if '<video>' not in first_msg.get('value', ''):
                    errors.append(f"Missing <video> tag in {split_name}[{index}] first message")

    return errors

def validate_file(file_path, split_name):
    """Validate a single JSON file."""
    print(f"\n{'='*60}")
    print(f"Validating {split_name}: {file_path}")
    print('='*60)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False

    if not isinstance(data, list):
        print(f"❌ Data must be a list, got {type(data)}")
        return False

    print(f"✓ Total samples: {len(data)}")

    # Validate samples
    all_errors = []
    for i, sample in enumerate(data):
        errors = validate_sample(sample, split_name, i)
        if errors:
            all_errors.extend(errors)

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} error(s):")
        for error in all_errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
        return False

    # Show sample
    if len(data) > 0:
        print("\n✓ Format validation passed!")
        print("\nSample data (first entry):")
        print("-" * 60)
        sample = data[0]
        print(f"Video: {sample.get('video', 'N/A')}")
        print(f"Conversations: {len(sample.get('conversations', []))} turns")
        if 'conversations' in sample and len(sample['conversations']) > 0:
            print(f"\nFirst message:")
            first_conv = sample['conversations'][0]
            print(f"  From: {first_conv.get('from', 'N/A')}")
            print(f"  Value (first 100 chars): {first_conv.get('value', 'N/A')[:100]}...")
        print("-" * 60)

    return True

def main():
    """Main validation function."""
    print("\n" + "="*60)
    print("VideoLLaMA3 Dataset Format Validator")
    print("="*60)

    base_dir = Path("dataset")

    if not base_dir.exists():
        print(f"\n❌ Dataset directory not found: {base_dir}")
        print("Please run: bash scripts/setup/initialize_dataset.sh")
        sys.exit(1)

    files_to_check = {
        'train': base_dir / "qved_train.json",
        'val': base_dir / "qved_val.json",
        'test': base_dir / "qved_test.json"
    }

    results = {}
    for split_name, file_path in files_to_check.items():
        results[split_name] = validate_file(file_path, split_name)

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    all_passed = all(results.values())

    for split_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{split_name.upper():8} : {status}")

    print("="*60)

    if all_passed:
        print("\n✅ All validations passed!")
        print("\nDataset is ready for VideoLLaMA3 fine-tuning.")
        print("Run: bash scripts/finetune/finetune_qved.sh")
        return 0
    else:
        print("\n❌ Some validations failed!")
        print("Please check the errors above and fix the dataset.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
