"""
Meaning-preserving and meaning-breaking augmentations for the synthetic dataset.
"""

import re
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from dataset_generator import (
    SceneGraph, ColoredBlock, SpatialRelation,
    Color, Shape, Relation, TextRenderer
)


class AugmentationType(Enum):
    """Types of augmentations available"""
    # Meaning-preserving
    CONJUNCT_SHUFFLE = "conjunct_shuffle"
    SYNONYM_SWAP = "synonym_swap"
    PASSIVE_ACTIVE = "passive_active"
    DETERMINER_VARIATION = "determiner_variation"
    OBJECT_RENAME = "object_rename"
    ADD_REDUNDANCY = "add_redundancy"
    WORD_ORDER = "word_order"

    # Meaning-breaking (counterfactuals)
    COLOR_FLIP = "color_flip"
    RELATION_FLIP = "relation_flip"
    ARGUMENT_SWAP = "argument_swap"
    COUNT_CHANGE = "count_change"
    NEGATION = "negation"
    OBJECT_SUBSTITUTION = "object_substitution"


class MeaningPreservingAugmenter:
    """Generates paraphrases that preserve semantic meaning"""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        # Synonym mappings for relations
        self.relation_synonyms = {
            "on": ["on", "on top of", "sitting on", "placed on", "resting on", "atop"],
            "on top of": ["on top of", "on", "atop", "above", "sitting on"],
            "under": ["under", "beneath", "below", "underneath"],
            "below": ["below", "beneath", "under"],
            "beneath": ["beneath", "under", "below", "underneath"],
            "left of": ["to the left of", "left of", "on the left side of", "on the left of"],
            "right of": ["to the right of", "right of", "on the right side of", "on the right of"],
            "next to": ["next to", "beside", "adjacent to", "alongside"],
            "beside": ["beside", "next to", "adjacent to", "by"],
            "near": ["near", "close to", "nearby", "in proximity to"],
        }

        # Synonym mappings for shapes
        self.shape_synonyms = {
            "block": ["block", "cube", "brick"],
            "cube": ["cube", "block", "box"],
            "box": ["box", "container", "cube"],
        }

        # Color adjective variations
        self.color_modifiers = {
            "red": ["red", "crimson", "scarlet", "ruby"],
            "blue": ["blue", "azure", "cobalt", "sapphire"],
            "green": ["green", "emerald", "jade", "forest"],
            "yellow": ["yellow", "golden", "amber", "lemon"],
            "orange": ["orange", "tangerine", "coral"],
            "purple": ["purple", "violet", "lavender", "plum"],
        }

        # Determiners
        self.determiners = ["the", "a", "this", "that", "one"]

    def conjunct_shuffle(self, text: str) -> List[str]:
        """
        Shuffle conjuncts in compound sentences.
        'A and B' -> 'B and A'
        'A is on B and next to C' -> 'A is next to C and on B'
        """
        variants = []

        # Pattern for "X and Y" at sentence level
        and_pattern = r'(.+?)\s+and\s+(.+?)(?:\.|$)'
        match = re.search(and_pattern, text)

        if match:
            part1, part2 = match.groups()
            # Simple shuffle
            shuffled = f"{part2.strip()} and {part1.strip()}"
            if text.endswith('.'):
                shuffled += '.'
            variants.append(shuffled)

            # Also try to find subject and shuffle just the predicates
            # Pattern: "subject is/sits relation1 object1 and relation2 object2"
            subj_pattern = r'^(.*?)\s+(is|sits|is placed)\s+(.+?)\s+and\s+(.+?)(?:\.|$)'
            subj_match = re.match(subj_pattern, text)

            if subj_match:
                subject, verb, pred1, pred2 = subj_match.groups()
                shuffled_pred = f"{subject} {verb} {pred2.strip()} and {pred1.strip()}"
                if text.endswith('.'):
                    shuffled_pred += '.'
                variants.append(shuffled_pred)

        # Pattern for comma-separated items
        comma_pattern = r'(.+?),\s*(.+?),\s*and\s+(.+?)(?:\.|$)'
        match = re.search(comma_pattern, text)

        if match:
            items = [match.group(1), match.group(2), match.group(3)]
            # Generate a few shuffles
            random.shuffle(items)
            shuffled = f"{items[0]}, {items[1]}, and {items[2]}"
            if text.endswith('.'):
                shuffled += '.'
            variants.append(shuffled)

        return variants if variants else [text]

    def synonym_swap(self, text: str) -> List[str]:
        """
        Swap words with their synonyms.
        'on' -> 'on top of', 'block' -> 'cube', etc.
        """
        variants = []

        # Try swapping relation phrases
        for original, synonyms in self.relation_synonyms.items():
            # Create regex pattern that matches the relation as a whole word/phrase
            pattern = r'\b' + re.escape(original) + r'\b'

            if re.search(pattern, text):
                # Generate variants with different synonyms
                for synonym in synonyms:
                    if synonym != original:
                        variant = re.sub(pattern, synonym, text)
                        if variant != text:
                            variants.append(variant)
                            # Only do a few to avoid explosion
                            if len(variants) >= 3:
                                break

        # Try swapping shapes
        for original, synonyms in self.shape_synonyms.items():
            pattern = r'\b' + re.escape(original) + r'\b'

            if re.search(pattern, text):
                for synonym in synonyms:
                    if synonym != original:
                        variant = re.sub(pattern, synonym, text)
                        if variant != text and variant not in variants:
                            variants.append(variant)
                            if len(variants) >= 5:
                                break

        return variants[:5] if variants else [text]

    def passive_active_transform(self, text: str) -> List[str]:
        """
        Transform between passive and active voice.
        'A is supported by B' <-> 'B supports A'
        'The red block is on the blue cube' -> 'The blue cube supports the red block'
        """
        variants = []

        # Pattern: "X is [relation] Y" -> "Y [inverse_verb] X"
        # For "on/under" relations
        patterns = [
            (r'(the \w+ \w+) is on (the \w+ \w+)', r'\2 supports \1'),
            (r'(the \w+ \w+) is on top of (the \w+ \w+)', r'\2 supports \1'),
            (r'(the \w+ \w+) is under (the \w+ \w+)', r'\2 is above \1'),
            (r'(the \w+ \w+) is beneath (the \w+ \w+)', r'\2 is on top of \1'),
            (r'(the \w+ \w+) sits on (the \w+ \w+)', r'\2 holds \1'),
            (r'(the \w+ \w+) is placed on (the \w+ \w+)', r'\2 has \1 on it'),
        ]

        for pattern, replacement in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                variant = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                variants.append(variant)

        # Also try the reverse transformations
        reverse_patterns = [
            (r'(the \w+ \w+) supports (the \w+ \w+)', r'\2 is on \1'),
            (r'(the \w+ \w+) holds (the \w+ \w+)', r'\2 sits on \1'),
            (r'(the \w+ \w+) has (the \w+ \w+) on it', r'\2 is placed on \1'),
        ]

        for pattern, replacement in reverse_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                variant = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if variant not in variants:
                    variants.append(variant)

        return variants if variants else []

    def determiner_variation(self, text: str) -> List[str]:
        """
        Vary determiners while preserving meaning.
        'the red block' -> 'a red block', 'this red block', etc.
        """
        variants = []

        # Pattern to match determiner + adjective + noun
        det_pattern = r'\b(the|a|an|this|that|one)\s+(\w+\s+\w+)'

        matches = list(re.finditer(det_pattern, text))

        if matches:
            # Try different determiner combinations
            for _ in range(3):
                variant = text
                for match in matches:
                    original_det = match.group(1)
                    new_det = random.choice([d for d in self.determiners if d != original_det])
                    variant = variant.replace(match.group(0), f"{new_det} {match.group(2)}", 1)

                if variant != text and variant not in variants:
                    variants.append(variant)

        return variants[:3] if variants else []

    def object_rename(self, text: str, scene: Optional[SceneGraph] = None) -> List[str]:
        """
        Consistently rename object colors/properties while maintaining relations.
        'red block' -> 'crimson block' throughout the sentence
        """
        variants = []

        # Extract color-shape pairs
        color_shape_pattern = r'(the |a |this |that |one )?(red|blue|green|yellow|orange|purple|black|white|gray|brown)\s+(block|cube|box)'

        matches = list(re.finditer(color_shape_pattern, text, re.IGNORECASE))

        if matches:
            # Track which colors are present
            colors_in_text = set()
            for match in matches:
                colors_in_text.add(match.group(2).lower())

            # Try renaming each color consistently
            for color in colors_in_text:
                if color in self.color_modifiers:
                    alternatives = self.color_modifiers[color]
                    for alt_color in alternatives:
                        if alt_color != color:
                            # Replace all instances of this color
                            variant = re.sub(
                                r'\b' + color + r'\b',
                                alt_color,
                                text,
                                flags=re.IGNORECASE
                            )
                            if variant != text and variant not in variants:
                                variants.append(variant)
                                if len(variants) >= 3:
                                    break

        return variants[:3] if variants else []

    def add_redundancy(self, text: str) -> List[str]:
        """
        Add redundant but natural descriptions.
        'the red block' -> 'the red colored block'
        'on the cube' -> 'on top of the cube'
        """
        variants = []

        # Add "colored" after colors
        color_pattern = r'(the |a |this |that )(red|blue|green|yellow|orange|purple|black|white|gray|brown)\s+(block|cube|box)'

        match = re.search(color_pattern, text, re.IGNORECASE)
        if match:
            variant = re.sub(
                color_pattern,
                r'\1\2 colored \3',
                text,
                count=1,
                flags=re.IGNORECASE
            )
            variants.append(variant)

        # Elaborate on positions
        elaborations = [
            (r'\bon\b', 'on top of'),
            (r'\bnext to\b', 'right next to'),
            (r'\bnear\b', 'very near'),
            (r'is (on|under|near)', r'is positioned \1'),
            (r'sits', 'is sitting'),
        ]

        for pattern, replacement in elaborations:
            if re.search(pattern, text):
                variant = re.sub(pattern, replacement, text, count=1)
                if variant != text and variant not in variants:
                    variants.append(variant)

        return variants[:3] if variants else []

    def word_order_variation(self, text: str) -> List[str]:
        """
        Vary word order while preserving meaning.
        'The red block is on the blue cube' -> 'On the blue cube is the red block'
        """
        variants = []

        # Pattern: "X is [relation] Y" -> "[Relation] Y is X"
        patterns = [
            (r'^(the \w+ \w+) is (on|under|near|next to|beside) (the \w+ \w+)$',
             lambda m: f"{m.group(2).capitalize()} {m.group(3)} is {m.group(1)}"),
            (r'^(the \w+ \w+) is (on top of|to the left of|to the right of) (the \w+ \w+)$',
             lambda m: f"{m.group(2).capitalize()} {m.group(3)} is {m.group(1)}"),
        ]

        for pattern, transformer in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                variant = transformer(match)
                if text.endswith('.'):
                    variant += '.'
                variants.append(variant)

        # "There is X [relation] Y" variations
        there_pattern = r'^there is (.*?) (on|under|near|next to) (.*?)$'
        match = re.match(there_pattern, text, re.IGNORECASE)
        if match:
            obj1, rel, obj2 = match.groups()
            variants.append(f"{obj1} is {rel} {obj2}")
            variants.append(f"{obj1} can be found {rel} {obj2}")

        return variants[:2] if variants else []

    def generate_paraphrases(self, text: str, scene: Optional[SceneGraph] = None,
                           num_paraphrases: int = 5) -> List[str]:
        """
        Generate multiple paraphrases using various augmentation techniques.
        """
        all_variants = set([text])  # Start with original

        # Apply each augmentation type
        augmentations = [
            self.conjunct_shuffle,
            self.synonym_swap,
            self.passive_active_transform,
            self.determiner_variation,
            lambda t: self.object_rename(t, scene),
            self.add_redundancy,
            self.word_order_variation,
        ]

        # First pass: apply each augmentation to original
        for aug_func in augmentations:
            variants = aug_func(text)
            all_variants.update(variants)

        # Second pass: apply augmentations to some variants for more diversity
        current_variants = list(all_variants)
        for variant in current_variants[:5]:  # Limit to avoid explosion
            if variant != text:
                for aug_func in random.sample(augmentations, k=2):
                    new_variants = aug_func(variant)
                    all_variants.update(new_variants)
                    if len(all_variants) >= num_paraphrases * 2:
                        break

        # Remove the original text and any empty strings
        all_variants.discard(text)
        all_variants = [v for v in all_variants if v and v.strip()]

        # Return requested number of paraphrases
        if len(all_variants) >= num_paraphrases:
            return random.sample(list(all_variants), num_paraphrases)
        else:
            return list(all_variants)


class CounterfactualGenerator:
    """Generates counterfactual (meaning-breaking) variations"""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.colors = ["red", "blue", "green", "yellow", "orange", "purple",
                      "black", "white", "gray", "brown"]
        self.shapes = ["block", "cube", "box"]
        self.relations = ["on", "under", "left of", "right of", "next to",
                         "near", "beside", "on top of", "beneath", "below"]

        # Opposite relations for flipping
        self.opposite_relations = {
            "on": "under",
            "on top of": "beneath",
            "above": "below",
            "under": "on",
            "below": "above",
            "beneath": "on top of",
            "left of": "right of",
            "right of": "left of",
            "near": "far from",
            "far from": "near",
        }

    def flip_color(self, text: str) -> List[str]:
        """
        Change colors to create different scenes.
        'red block on blue cube' -> 'green block on blue cube'
        """
        variants = []

        # Find all colors in the text
        color_pattern = r'\b(' + '|'.join(self.colors) + r')\b'
        colors_found = re.findall(color_pattern, text, re.IGNORECASE)

        if colors_found:
            # For each color found, try replacing with others
            for original_color in set(colors_found):
                # Pick a different color
                other_colors = [c for c in self.colors if c != original_color.lower()]

                for new_color in random.sample(other_colors, min(3, len(other_colors))):
                    variant = re.sub(
                        r'\b' + original_color + r'\b',
                        new_color,
                        text,
                        flags=re.IGNORECASE
                    )
                    if variant != text:
                        variants.append(variant)

            # Also try swapping two colors if there are multiple
            if len(set(colors_found)) >= 2:
                color_list = list(set(colors_found))
                color1, color2 = color_list[0], color_list[1]
                # Swap them
                temp_placeholder = "TEMP_COLOR_PLACEHOLDER"
                variant = text
                variant = re.sub(r'\b' + color1 + r'\b', temp_placeholder, variant, flags=re.IGNORECASE)
                variant = re.sub(r'\b' + color2 + r'\b', color1, variant, flags=re.IGNORECASE)
                variant = re.sub(temp_placeholder, color2, variant)
                if variant != text:
                    variants.append(variant)

        return variants[:5] if variants else []

    def flip_relation(self, text: str) -> List[str]:
        """
        Change spatial relations to create different configurations.
        'A on B' -> 'A under B', 'A left of B'
        """
        variants = []

        # Find relations in text
        for original_rel, opposite_rel in self.opposite_relations.items():
            pattern = r'\b' + re.escape(original_rel) + r'\b'

            if re.search(pattern, text, re.IGNORECASE):
                # Replace with opposite
                variant = re.sub(pattern, opposite_rel, text, flags=re.IGNORECASE)
                if variant != text:
                    variants.append(variant)

                # Also try other non-opposite relations
                other_rels = [r for r in self.relations
                            if r not in [original_rel, opposite_rel]]

                for other_rel in random.sample(other_rels, min(2, len(other_rels))):
                    variant = re.sub(pattern, other_rel, text, flags=re.IGNORECASE)
                    if variant != text and variant not in variants:
                        variants.append(variant)

        return variants[:4] if variants else []

    def swap_arguments(self, text: str) -> List[str]:
        """
        Swap the subject and object in relations.
        'A on B' -> 'B on A'
        """
        variants = []

        # Pattern: "X is [relation] Y"
        patterns = [
            r'(the \w+ \w+) (is|sits|is placed) (on|under|near|beside|next to|on top of|beneath) (the \w+ \w+)',
            r'(the \w+ \w+) (is|sits) (to the left of|to the right of|left of|right of) (the \w+ \w+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                subj, verb, relation, obj = match.groups()
                # Swap subject and object
                variant = text.replace(match.group(0), f"{obj} {verb} {relation} {subj}")
                if variant != text:
                    variants.append(variant)

        # Pattern: "There is X [relation] Y"
        there_pattern = r'there is (.*?) (on|under|near|next to|beside) (.*?)(?:\.|$)'
        match = re.search(there_pattern, text, re.IGNORECASE)
        if match:
            obj1, rel, obj2 = match.groups()
            variant = f"there is {obj2} {rel} {obj1}"
            if text.endswith('.'):
                variant += '.'
            variants.append(variant)

        return variants[:2] if variants else []

    def change_count(self, text: str) -> List[str]:
        """
        Change quantities to create different scenes.
        'a block' -> 'two blocks', 'several blocks'
        """
        variants = []

        # Simple number additions
        quantifiers = ["two", "three", "several", "many", "multiple"]

        # Pattern for single objects
        single_pattern = r'\b(a|an|the|one)\s+(\w+)\s+(block|cube|box)\b'

        matches = re.finditer(single_pattern, text, re.IGNORECASE)

        for match in matches:
            det, color, shape = match.groups()

            # Make plural and add quantifier
            plural_shape = shape + "s"

            for quant in random.sample(quantifiers, min(2, len(quantifiers))):
                variant = text.replace(
                    match.group(0),
                    f"{quant} {color} {plural_shape}"
                )
                if variant != text and variant not in variants:
                    variants.append(variant)

        return variants[:3] if variants else []

    def add_negation(self, text: str) -> List[str]:
        """
        Add negation to create false statements.
        'A is on B' -> 'A is not on B'
        """
        variants = []

        # Pattern for "is [relation]"
        is_pattern = r'\b(is|sits)\s+(on|under|near|beside|next to|on top of|beneath|left of|right of)'

        match = re.search(is_pattern, text, re.IGNORECASE)
        if match:
            verb = match.group(1)
            # Add "not" after the verb
            negated = text.replace(match.group(0), f"{verb} not {match.group(2)}")
            variants.append(negated)

            # Also try "is no longer"
            no_longer = text.replace(match.group(0), f"is no longer {match.group(2)}")
            variants.append(no_longer)

        # Pattern for "there is"
        if text.startswith("there is"):
            variants.append(text.replace("there is", "there is no"))
            variants.append(text.replace("there is", "there isn't"))

        return variants[:2] if variants else []

    def object_substitution(self, text: str) -> List[str]:
        """
        Replace objects with completely different ones.
        'red block' -> 'red sphere', 'blue pyramid'
        """
        variants = []

        # Extended shapes for substitution
        alt_shapes = ["sphere", "pyramid", "cylinder", "ball", "disk", "rod"]

        # Pattern for color + shape
        shape_pattern = r'(\w+)\s+(block|cube|box)'

        matches = re.finditer(shape_pattern, text, re.IGNORECASE)

        for match in matches:
            color = match.group(1)
            original_shape = match.group(2)

            # Substitute with alternative shapes
            for alt_shape in random.sample(alt_shapes, min(2, len(alt_shapes))):
                variant = text.replace(match.group(0), f"{color} {alt_shape}")
                if variant != text and variant not in variants:
                    variants.append(variant)

            # Also try different color + different shape
            for alt_color in random.sample(self.colors, 2):
                if alt_color != color.lower():
                    alt_shape = random.choice(alt_shapes)
                    variant = text.replace(match.group(0), f"{alt_color} {alt_shape}")
                    if variant != text and variant not in variants:
                        variants.append(variant)

        return variants[:4] if variants else []

    def generate_counterfactuals(self, text: str, num_counterfactuals: int = 5) -> List[str]:
        """
        Generate multiple counterfactuals using various techniques.
        """
        all_counterfactuals = []

        # Apply each counterfactual generation method
        methods = [
            self.flip_color,
            self.flip_relation,
            self.swap_arguments,
            self.change_count,
            self.add_negation,
            self.object_substitution,
        ]

        for method in methods:
            counterfactuals = method(text)
            all_counterfactuals.extend(counterfactuals)

        # Remove duplicates while preserving order
        seen = set()
        unique_counterfactuals = []
        for cf in all_counterfactuals:
            if cf not in seen and cf != text:
                seen.add(cf)
                unique_counterfactuals.append(cf)

        # Return requested number
        if len(unique_counterfactuals) >= num_counterfactuals:
            return random.sample(unique_counterfactuals, num_counterfactuals)
        else:
            return unique_counterfactuals


def main():
    """Test augmentation functions"""

    print("=" * 60)
    print("Testing Meaning-Preserving Augmentations")
    print("=" * 60)

    augmenter = MeaningPreservingAugmenter(seed=42)

    test_sentences = [
        "the red block is on the blue cube",
        "the green box is next to the yellow block and under the purple cube",
        "there is a red block on the blue cube",
        "the small red cube sits beside the large green box",
    ]

    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        print("-" * 40)

        # Test individual augmentations
        print("Conjunct shuffle:", augmenter.conjunct_shuffle(sentence))
        print("Synonym swap:", augmenter.synonym_swap(sentence)[:2])
        print("Passive/Active:", augmenter.passive_active_transform(sentence))
        print("Determiner variation:", augmenter.determiner_variation(sentence)[:2])
        print("Object rename:", augmenter.object_rename(sentence)[:2])
        print("Add redundancy:", augmenter.add_redundancy(sentence)[:2])
        print("Word order:", augmenter.word_order_variation(sentence))

        print("\nCombined paraphrases:")
        paraphrases = augmenter.generate_paraphrases(sentence, num_paraphrases=5)
        for i, para in enumerate(paraphrases, 1):
            print(f"  {i}. {para}")

    print("\n" + "=" * 60)
    print("Testing Counterfactual Generation")
    print("=" * 60)

    counter_gen = CounterfactualGenerator(seed=42)

    for sentence in test_sentences[:2]:
        print(f"\nOriginal: {sentence}")
        print("-" * 40)

        # Test individual counterfactual methods
        print("Color flip:", counter_gen.flip_color(sentence)[:2])
        print("Relation flip:", counter_gen.flip_relation(sentence)[:2])
        print("Argument swap:", counter_gen.swap_arguments(sentence))
        print("Count change:", counter_gen.change_count(sentence)[:2])
        print("Negation:", counter_gen.add_negation(sentence))
        print("Object substitution:", counter_gen.object_substitution(sentence)[:2])

        print("\nCombined counterfactuals:")
        counterfactuals = counter_gen.generate_counterfactuals(sentence, num_counterfactuals=5)
        for i, cf in enumerate(counterfactuals, 1):
            print(f"  {i}. {cf}")


if __name__ == "__main__":
    main()