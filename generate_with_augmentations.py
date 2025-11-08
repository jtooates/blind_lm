#!/usr/bin/env python3
"""
Generate sentences with their paraphrases and counterfactuals.
Shows meaning-preserving augmentations and meaning-breaking variations.

Usage:
    python generate_with_augmentations.py --num 5
    python generate_with_augmentations.py --num 10 --paraphrases 3 --counterfactuals 3
    python generate_with_augmentations.py --format json --output dataset.jsonl
"""

import argparse
import json
import sys
from dataset_generator import SceneGenerator, TextRenderer
from augmentations import MeaningPreservingAugmenter, CounterfactualGenerator


def generate_with_augmentations(num_scenes, complexity, seed, num_paraphrases,
                                num_counterfactuals, output_format, output_file):
    """
    Generate scenes with paraphrases and counterfactuals.

    Args:
        num_scenes: Number of scenes to generate
        complexity: Scene complexity level (1, 2, or 3)
        seed: Random seed for reproducibility
        num_paraphrases: Number of paraphrases per scene
        num_counterfactuals: Number of counterfactuals per scene
        output_format: 'text' or 'json' or 'jsonl'
        output_file: File path to write to, or None for stdout
    """
    # Initialize components
    scene_gen = SceneGenerator(seed=seed)
    renderer = TextRenderer()
    para_augmenter = MeaningPreservingAugmenter(seed=seed)
    counter_gen = CounterfactualGenerator(seed=seed)

    # Open output stream
    if output_file:
        f = open(output_file, 'w')
    else:
        f = sys.stdout

    try:
        for i in range(num_scenes):
            # Generate scene
            scene = scene_gen.generate_connected_scene(complexity)

            # Validate
            is_valid, errors = scene.validate()
            if not is_valid:
                continue

            # Render original text
            texts = renderer.render_simple(scene, use_variety=False)
            if not texts:
                continue

            original_text = texts[0]

            # Generate paraphrases (meaning-preserving)
            paraphrases = para_augmenter.generate_paraphrases(
                original_text,
                scene=scene,
                num_paraphrases=num_paraphrases
            )

            # Generate counterfactuals (meaning-breaking)
            counterfactuals = counter_gen.generate_counterfactuals(
                original_text,
                num_counterfactuals=num_counterfactuals
            )

            # Output based on format
            if output_format == 'text':
                output_text_format(f, i, original_text, paraphrases, counterfactuals)
            elif output_format == 'json':
                output_json_format(f, i, scene, original_text, paraphrases,
                                 counterfactuals, complexity, is_last=(i == num_scenes - 1))
            elif output_format == 'jsonl':
                output_jsonl_format(f, i, scene, original_text, paraphrases,
                                  counterfactuals, complexity)

            # Progress to stderr if writing to file
            if output_file and (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_scenes} scenes...", file=sys.stderr)

    finally:
        if output_file:
            f.close()
            print(f"Wrote {num_scenes} scenes to {output_file}", file=sys.stderr)


def output_text_format(f, idx, original, paraphrases, counterfactuals):
    """Output in human-readable text format"""
    f.write(f"\n{'='*70}\n")
    f.write(f"Scene {idx + 1}\n")
    f.write(f"{'='*70}\n")
    f.write(f"\nOriginal:\n  {original}\n")

    f.write(f"\nParaphrases (meaning-preserving):\n")
    for i, para in enumerate(paraphrases, 1):
        f.write(f"  {i}. {para}\n")

    f.write(f"\nCounterfactuals (meaning-breaking):\n")
    for i, cf in enumerate(counterfactuals, 1):
        f.write(f"  {i}. {cf}\n")


def output_json_format(f, idx, scene, original, paraphrases, counterfactuals, complexity, is_last):
    """Output in JSON array format"""
    if idx == 0:
        f.write("[\n")

    data = {
        "id": f"scene_{scene.get_scene_hash()}",
        "scene_graph": scene.to_dict(),
        "complexity": complexity,
        "original": original,
        "paraphrases": paraphrases,
        "counterfactuals": counterfactuals
    }

    json_str = json.dumps(data, indent=2)

    # Indent each line for array formatting
    lines = json_str.split('\n')
    indented = '\n'.join('  ' + line for line in lines)
    f.write(indented)

    if not is_last:
        f.write(",\n")
    else:
        f.write("\n]\n")


def output_jsonl_format(f, idx, scene, original, paraphrases, counterfactuals, complexity):
    """Output in JSONL format (one JSON object per line)"""
    data = {
        "id": f"scene_{scene.get_scene_hash()}",
        "scene_graph": scene.to_dict(),
        "complexity": complexity,
        "original": original,
        "paraphrases": paraphrases,
        "counterfactuals": counterfactuals
    }

    f.write(json.dumps(data) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Generate sentences with paraphrases and counterfactuals.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--num', '-n',
        type=int,
        default=5,
        help='Number of scenes to generate'
    )

    parser.add_argument(
        '--complexity', '-c',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='Scene complexity level (1=simple, 2=medium, 3=complex)'
    )

    parser.add_argument(
        '--paraphrases', '-p',
        type=int,
        default=5,
        help='Number of paraphrases per scene'
    )

    parser.add_argument(
        '--counterfactuals', '-cf',
        type=int,
        default=5,
        help='Number of counterfactuals per scene'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['text', 'json', 'jsonl'],
        default='text',
        help='Output format (text=human-readable, json=array, jsonl=streaming)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: stdout)'
    )

    args = parser.parse_args()

    # Generate
    generate_with_augmentations(
        num_scenes=args.num,
        complexity=args.complexity,
        seed=args.seed,
        num_paraphrases=args.paraphrases,
        num_counterfactuals=args.counterfactuals,
        output_format=args.format,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
