"""
Test script for the dataset generator.
Verifies core functionality and demonstrates usage.
"""

import json
from dataset_generator import (
    SceneGenerator, TextRenderer, SceneGraph,
    ColoredBlock, SpatialRelation, Color, Shape, Relation
)


def test_basic_scene_creation():
    """Test creating a simple scene manually"""
    print("Test 1: Manual Scene Creation")
    print("-" * 40)

    scene = SceneGraph()

    # Create objects
    red_block = ColoredBlock("obj_001", Color.RED, Shape.BLOCK)
    blue_cube = ColoredBlock("obj_002", Color.BLUE, Shape.CUBE)

    scene.add_object(red_block)
    scene.add_object(blue_cube)

    # Add relation
    relation = SpatialRelation(red_block, Relation.ON, blue_cube)
    scene.add_relation(relation)

    # Validate
    is_valid, errors = scene.validate()
    print(f"Scene valid: {is_valid}")
    print(f"Scene hash: {scene.get_scene_hash()}")
    print(f"Canonical: {scene.to_canonical()}")

    # Render
    renderer = TextRenderer()
    sentences = renderer.render_simple(scene)
    print("Rendered sentences:")
    for s in sentences:
        print(f"  - {s}")

    print()


def test_scene_generation():
    """Test automatic scene generation"""
    print("Test 2: Automatic Scene Generation")
    print("-" * 40)

    generator = SceneGenerator(seed=42)

    for complexity in [1, 2, 3]:
        print(f"\nComplexity {complexity}:")
        scene = generator.generate_connected_scene(complexity)

        print(f"  Objects: {len(scene.objects)}")
        print(f"  Relations: {len(scene.relations)}")

        is_valid, errors = scene.validate()
        print(f"  Valid: {is_valid}")

        if not is_valid:
            for error in errors:
                print(f"    ERROR: {error}")

    print()


def test_validation():
    """Test scene validation catches invalid configurations"""
    print("Test 3: Scene Validation")
    print("-" * 40)

    scene = SceneGraph()

    # Create cycle: A on B, B on C, C on A
    obj_a = ColoredBlock("A", Color.RED, Shape.BLOCK)
    obj_b = ColoredBlock("B", Color.BLUE, Shape.CUBE)
    obj_c = ColoredBlock("C", Color.GREEN, Shape.BOX)

    scene.add_object(obj_a)
    scene.add_object(obj_b)
    scene.add_object(obj_c)

    scene.add_relation(SpatialRelation(obj_a, Relation.ON, obj_b))
    scene.add_relation(SpatialRelation(obj_b, Relation.ON, obj_c))
    scene.add_relation(SpatialRelation(obj_c, Relation.ON, obj_a))

    is_valid, errors = scene.validate()
    print(f"Cycle scene valid: {is_valid}")
    print("Errors detected:")
    for error in errors:
        print(f"  - {error}")

    # Test contradictory relations
    scene2 = SceneGraph()
    scene2.add_object(obj_a)
    scene2.add_object(obj_b)
    scene2.add_relation(SpatialRelation(obj_a, Relation.LEFT_OF, obj_b))
    scene2.add_relation(SpatialRelation(obj_a, Relation.RIGHT_OF, obj_b))

    is_valid, errors = scene2.validate()
    print(f"\nContradiction scene valid: {is_valid}")
    print("Errors detected:")
    for error in errors:
        print(f"  - {error}")

    print()


def test_text_variety():
    """Test different text rendering options"""
    print("Test 4: Text Rendering Variety")
    print("-" * 40)

    generator = SceneGenerator(seed=42)
    renderer = TextRenderer()

    scene = generator.generate_connected_scene(complexity=2)

    print("Scene relations:")
    for rel in scene.relations:
        print(f"  - {rel.subject.color.value} {rel.subject.shape.value} "
              f"{rel.relation.value} "
              f"{rel.object.color.value} {rel.object.shape.value}")

    print("\nSimple (no variety):")
    for s in renderer.render_simple(scene, use_variety=False)[:3]:
        print(f"  - {s}")

    print("\nSimple (with variety):")
    for s in renderer.render_simple(scene, use_variety=True)[:3]:
        print(f"  - {s}")

    print("\nComplex sentences:")
    for s in renderer.render_complex(scene)[:3]:
        print(f"  - {s}")

    print()


def test_inverse_relations():
    """Test inverse relation generation"""
    print("Test 5: Inverse Relations")
    print("-" * 40)

    red = ColoredBlock("red", Color.RED, Shape.BLOCK)
    blue = ColoredBlock("blue", Color.BLUE, Shape.CUBE)

    relation = SpatialRelation(red, Relation.ON, blue)
    inverse = relation.inverse()

    print(f"Original: {red.color.value} {relation.relation.value} {blue.color.value}")
    print(f"Inverse: {inverse.subject.color.value} {inverse.relation.value} {inverse.object.color.value}")

    # Test all inversible relations
    print("\nAll inverse pairs:")
    for rel in [Relation.ON, Relation.LEFT_OF, Relation.ABOVE]:
        test_rel = SpatialRelation(red, rel, blue)
        inv = test_rel.inverse()
        if inv:
            print(f"  {rel.value} <-> {inv.relation.value}")

    print()


def test_scene_hashing():
    """Test that identical scenes produce same hash"""
    print("Test 6: Scene Hashing and Canonical Form")
    print("-" * 40)

    # Create two identical scenes with different object order
    scene1 = SceneGraph()
    scene2 = SceneGraph()

    red = ColoredBlock("obj_001", Color.RED, Shape.BLOCK)
    blue = ColoredBlock("obj_002", Color.BLUE, Shape.CUBE)

    # Add in different order
    scene1.add_object(red)
    scene1.add_object(blue)
    scene1.add_relation(SpatialRelation(red, Relation.ON, blue))

    scene2.add_object(blue)
    scene2.add_object(red)
    scene2.add_relation(SpatialRelation(red, Relation.ON, blue))

    print(f"Scene 1 hash: {scene1.get_scene_hash()}")
    print(f"Scene 2 hash: {scene2.get_scene_hash()}")
    print(f"Hashes match: {scene1.get_scene_hash() == scene2.get_scene_hash()}")

    print(f"\nCanonical 1: {scene1.to_canonical()}")
    print(f"Canonical 2: {scene2.to_canonical()}")

    print()


def test_json_serialization():
    """Test converting scenes to/from JSON"""
    print("Test 7: JSON Serialization")
    print("-" * 40)

    generator = SceneGenerator(seed=42)
    scene = generator.generate_connected_scene(complexity=1)

    # Convert to dict
    scene_dict = scene.to_dict()
    json_str = json.dumps(scene_dict, indent=2)

    print("JSON representation:")
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)

    # Could implement from_dict method to reconstruct
    print(f"\nScene has {len(scene_dict['objects'])} objects and {len(scene_dict['relations'])} relations")

    print()


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Dataset Generator Tests")
    print("=" * 50)
    print()

    test_basic_scene_creation()
    test_scene_generation()
    test_validation()
    test_text_variety()
    test_inverse_relations()
    test_scene_hashing()
    test_json_serialization()

    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()