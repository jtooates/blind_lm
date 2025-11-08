#!/usr/bin/env python3
"""
Utility script to generate raw sentences from synthetic scenes.
Outputs to stdout by default, or to a file if specified.

Usage:
    python generate_sentences.py --num 100 --complexity 1
    python generate_sentences.py --num 1000 --complexity 2 --variety --output sentences.txt
    python generate_sentences.py --help
"""

import argparse
import sys
from dataset_generator import SceneGenerator, TextRenderer


def generate_sentences(num_sentences, complexity, use_variety, seed, output_file, use_complex):
    """
    Generate synthetic sentences and output them.

    Args:
        num_sentences: Number of sentences to generate
        complexity: Scene complexity level (1, 2, or 3)
        use_variety: Whether to use varied phrasings
        seed: Random seed for reproducibility
        output_file: File path to write to, or None for stdout
        use_complex: Whether to use complex sentence rendering
    """
    # Initialize components
    generator = SceneGenerator(seed=seed)
    renderer = TextRenderer()

    # Open output stream
    if output_file:
        f = open(output_file, 'w')
    else:
        f = sys.stdout

    try:
        # Generate sentences
        for i in range(num_sentences):
            # Generate a scene
            scene = generator.generate_connected_scene(complexity=complexity)

            # Render to text based on rendering mode
            if use_complex:
                # Use complex rendering (conjunctions, relation chains)
                texts = renderer.render_complex(scene)
            else:
                # Use simple rendering
                texts = renderer.render_simple(scene, use_variety=use_variety)

            if texts:
                # Take the first rendering
                sentence = texts[0]
                f.write(sentence + '\n')

            # Progress indicator to stderr if writing to file
            if output_file and (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_sentences} sentences...", file=sys.stderr)

    finally:
        # Close file if we opened one
        if output_file:
            f.close()
            print(f"Wrote {num_sentences} sentences to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic sentences from colored block scenes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--num', '-n',
        type=int,
        default=10,
        help='Number of sentences to generate'
    )

    parser.add_argument(
        '--complexity', '-c',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='Scene complexity level (1=simple, 2=medium, 3=complex)'
    )

    parser.add_argument(
        '--variety', '-v',
        action='store_true',
        help='Use varied phrasings (determiners, synonyms, etc.)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: stdout)'
    )

    parser.add_argument(
        '--complex',
        action='store_true',
        help='Use complex sentence rendering with conjunctions and relation chains'
    )

    args = parser.parse_args()

    # Generate sentences
    generate_sentences(
        num_sentences=args.num,
        complexity=args.complexity,
        use_variety=args.variety,
        seed=args.seed,
        output_file=args.output,
        use_complex=args.complex
    )


if __name__ == '__main__':
    main()
