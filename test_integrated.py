"""
Integrated test showing scene generation with augmentations.
Demonstrates the full pipeline for Phase 0.
"""

import json
from dataset_generator import SceneGenerator, TextRenderer
from augmentations import MeaningPreservingAugmenter, CounterfactualGenerator


def test_integrated_pipeline():
    """Test the complete pipeline: scene -> text -> paraphrases -> counterfactuals"""

    print("=" * 70)
    print("INTEGRATED DATASET GENERATION PIPELINE")
    print("=" * 70)

    # Initialize components
    scene_gen = SceneGenerator(seed=42)
    text_renderer = TextRenderer()
    para_augmenter = MeaningPreservingAugmenter(seed=42)
    counter_gen = CounterfactualGenerator(seed=42)

    # Generate a few scenes at different complexity levels
    for complexity in [1, 2]:
        print(f"\n{'='*70}")
        print(f"COMPLEXITY LEVEL {complexity}")
        print('='*70)

        for scene_idx in range(2):
            # Generate scene
            scene = scene_gen.generate_connected_scene(complexity)

            print(f"\n--- Scene {scene_idx + 1} ---")
            print(f"Hash: {scene.get_scene_hash()}")
            print(f"Objects: {len(scene.objects)} | Relations: {len(scene.relations)}")

            # Validate scene
            is_valid, errors = scene.validate()
            if not is_valid:
                print(f"WARNING: Invalid scene - {errors}")
                continue

            # Render to text (multiple options)
            simple_texts = text_renderer.render_simple(scene, use_variety=False)
            varied_texts = text_renderer.render_simple(scene, use_variety=True)
            complex_texts = text_renderer.render_complex(scene) if complexity > 1 else []

            # Pick primary text
            primary_text = simple_texts[0] if simple_texts else ""

            print(f"\nPrimary text: {primary_text}")

            # Generate paraphrases
            paraphrases = para_augmenter.generate_paraphrases(
                primary_text,
                scene=scene,
                num_paraphrases=5
            )

            print("\nParaphrases (meaning-preserving):")
            for i, para in enumerate(paraphrases[:5], 1):
                print(f"  {i}. {para}")

            # Generate counterfactuals
            counterfactuals = counter_gen.generate_counterfactuals(
                primary_text,
                num_counterfactuals=5
            )

            print("\nCounterfactuals (meaning-breaking):")
            for i, cf in enumerate(counterfactuals[:5], 1):
                print(f"  {i}. {cf}")

            # Show data structure for export
            data_entry = {
                "id": f"scene_{complexity}_{scene_idx}_{scene.get_scene_hash()}",
                "scene_graph": scene.to_dict(),
                "complexity": complexity,
                "primary_text": primary_text,
                "alternative_renderings": varied_texts[:3] + (complex_texts[:2] if complex_texts else []),
                "paraphrases": paraphrases,
                "counterfactuals": counterfactuals,
            }

            print("\nExport format preview:")
            preview = {k: v if k != "scene_graph" else "..." for k, v in data_entry.items()}
            print(json.dumps(preview, indent=2)[:500] + "...")

    print(f"\n{'='*70}")
    print("PIPELINE TEST COMPLETE")
    print('='*70)


def test_semantic_verification():
    """Verify that paraphrases preserve meaning while counterfactuals don't"""

    print("\n" + "=" * 70)
    print("SEMANTIC VERIFICATION TEST")
    print("=" * 70)

    # Create a specific scene
    scene_gen = SceneGenerator(seed=100)
    scene = scene_gen.generate_connected_scene(complexity=1)

    text_renderer = TextRenderer()
    para_augmenter = MeaningPreservingAugmenter(seed=100)
    counter_gen = CounterfactualGenerator(seed=100)

    # Get canonical representation
    canonical = scene.to_canonical()

    # Render original
    original_text = text_renderer.render_simple(scene)[0]

    print(f"\nOriginal scene: {original_text}")
    print(f"Canonical form: {canonical[:50]}...")

    # Generate paraphrases
    paraphrases = para_augmenter.generate_paraphrases(original_text, scene, num_paraphrases=3)

    print("\n✓ Paraphrases should describe the SAME scene:")
    for para in paraphrases:
        print(f"  - {para}")
        # In a full implementation, we'd parse this back to verify it matches the canonical form

    # Generate counterfactuals
    counterfactuals = counter_gen.generate_counterfactuals(original_text, num_counterfactuals=3)

    print("\n✗ Counterfactuals should describe DIFFERENT scenes:")
    for cf in counterfactuals:
        print(f"  - {cf}")
        # These should parse to different canonical forms

    print("\nVerification approach:")
    print("  1. Parse each text back to scene graph (needs NLU component)")
    print("  2. Compare canonical forms")
    print("  3. Paraphrases should match original canonical")
    print("  4. Counterfactuals should NOT match original canonical")


def test_augmentation_diversity():
    """Test that we get sufficient diversity in augmentations"""

    print("\n" + "=" * 70)
    print("AUGMENTATION DIVERSITY TEST")
    print("=" * 70)

    test_text = "the red block is on the blue cube"

    para_augmenter = MeaningPreservingAugmenter()

    # Generate many paraphrases to check diversity
    all_paraphrases = set()
    for seed in range(10):
        para_augmenter = MeaningPreservingAugmenter(seed=seed)
        paras = para_augmenter.generate_paraphrases(test_text, num_paraphrases=10)
        all_paraphrases.update(paras)

    print(f"\nOriginal: {test_text}")
    print(f"Generated {len(all_paraphrases)} unique paraphrases across 10 seeds:")

    # Group by augmentation type detected
    categories = {
        "synonym": [],
        "determiner": [],
        "passive": [],
        "word_order": [],
        "redundancy": [],
        "other": []
    }

    for para in all_paraphrases:
        if "on top of" in para or "sitting on" in para or "placed on" in para:
            categories["synonym"].append(para)
        elif "supports" in para or "holds" in para:
            categories["passive"].append(para)
        elif para.startswith("On ") or para.startswith("There"):
            categories["word_order"].append(para)
        elif "colored" in para or "positioned" in para:
            categories["redundancy"].append(para)
        elif any(det in para for det in ["a ", "this ", "that ", "one "]):
            categories["determiner"].append(para)
        else:
            categories["other"].append(para)

    for category, examples in categories.items():
        if examples:
            print(f"\n{category.upper()} variations ({len(examples)}):")
            for ex in examples[:3]:
                print(f"  - {ex}")

    # Test counterfactual diversity
    counter_gen = CounterfactualGenerator()

    all_counterfactuals = set()
    for seed in range(10):
        counter_gen = CounterfactualGenerator(seed=seed)
        cfs = counter_gen.generate_counterfactuals(test_text, num_counterfactuals=10)
        all_counterfactuals.update(cfs)

    print(f"\n\nGenerated {len(all_counterfactuals)} unique counterfactuals across 10 seeds")

    cf_categories = {
        "color_change": [],
        "relation_change": [],
        "argument_swap": [],
        "negation": [],
        "object_change": [],
        "other": []
    }

    for cf in all_counterfactuals:
        if "not" in cf or "no longer" in cf:
            cf_categories["negation"].append(cf)
        elif "sphere" in cf or "pyramid" in cf or "cylinder" in cf:
            cf_categories["object_change"].append(cf)
        elif "blue cube is on the red" in cf:
            cf_categories["argument_swap"].append(cf)
        elif any(color in cf for color in ["green", "yellow", "purple", "orange"]) and "red" not in cf:
            cf_categories["color_change"].append(cf)
        elif any(rel in cf for rel in ["under", "next to", "beside", "near"]):
            cf_categories["relation_change"].append(cf)
        else:
            cf_categories["other"].append(cf)

    for category, examples in cf_categories.items():
        if examples:
            print(f"\n{category.upper()} counterfactuals ({len(examples)}):")
            for ex in examples[:3]:
                print(f"  - {ex}")


def main():
    """Run all integrated tests"""
    test_integrated_pipeline()
    test_semantic_verification()
    test_augmentation_diversity()


if __name__ == "__main__":
    main()