"""
Synthetic dataset generator for colored blocks and spatial relations.
Generates scenes with objects and relations, then renders them to text with paraphrases.
"""

import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import hashlib
from collections import defaultdict


class Color(Enum):
    """Available colors for objects"""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    BROWN = "brown"
    # Extended colors for later phases
    CYAN = "cyan"
    MAGENTA = "magenta"
    PINK = "pink"
    LIME = "lime"
    NAVY = "navy"
    TEAL = "teal"
    MAROON = "maroon"


class Shape(Enum):
    """Available shapes for objects"""
    BLOCK = "block"
    CUBE = "cube"
    BOX = "box"
    # Extended shapes for later phases
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    PYRAMID = "pyramid"
    BALL = "ball"


class Relation(Enum):
    """Spatial relations between objects"""
    # Vertical relations
    ON = "on"
    ON_TOP_OF = "on_top_of"
    ABOVE = "above"
    UNDER = "under"
    BELOW = "below"
    BENEATH = "beneath"
    # Horizontal relations
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    NEXT_TO = "next_to"
    BESIDE = "beside"
    # Proximity
    NEAR = "near"
    FAR_FROM = "far_from"
    TOUCHING = "touching"
    # Containment (for later phases)
    IN = "in"
    INSIDE = "inside"
    CONTAINS = "contains"


@dataclass
class ColoredBlock:
    """Represents a single colored object"""
    id: str
    color: Color
    shape: Shape
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z) for validation

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, ColoredBlock) and self.id == other.id

    def to_text(self, use_determiner: str = "the") -> str:
        """Convert to text representation"""
        return f"{use_determiner} {self.color.value} {self.shape.value}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "color": self.color.value,
            "shape": self.shape.value,
            "position": self.position
        }


@dataclass
class SpatialRelation:
    """Represents a spatial relationship between two objects"""
    subject: ColoredBlock
    relation: Relation
    object: ColoredBlock

    def inverse(self) -> Optional['SpatialRelation']:
        """Return the inverse relation if it exists"""
        inverse_map = {
            Relation.ON: Relation.UNDER,
            Relation.ON_TOP_OF: Relation.BENEATH,
            Relation.ABOVE: Relation.BELOW,
            Relation.UNDER: Relation.ON,
            Relation.BELOW: Relation.ABOVE,
            Relation.BENEATH: Relation.ON_TOP_OF,
            Relation.LEFT_OF: Relation.RIGHT_OF,
            Relation.RIGHT_OF: Relation.LEFT_OF,
        }
        if self.relation in inverse_map:
            return SpatialRelation(
                subject=self.object,
                relation=inverse_map[self.relation],
                object=self.subject
            )
        return None

    def to_dict(self) -> dict:
        return {
            "subject": self.subject.id,
            "relation": self.relation.value,
            "object": self.object.id
        }


@dataclass
class SceneGraph:
    """Represents the complete spatial arrangement of colored blocks"""
    objects: List[ColoredBlock] = field(default_factory=list)
    relations: List[SpatialRelation] = field(default_factory=list)

    def __post_init__(self):
        self.object_map = {obj.id: obj for obj in self.objects}

    def add_object(self, obj: ColoredBlock):
        """Add an object to the scene"""
        self.objects.append(obj)
        self.object_map[obj.id] = obj

    def add_relation(self, relation: SpatialRelation):
        """Add a relation to the scene"""
        self.relations.append(relation)

    def get_object_relations(self, obj: ColoredBlock) -> List[SpatialRelation]:
        """Get all relations involving an object"""
        return [r for r in self.relations
                if r.subject == obj or r.object == obj]

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate physical plausibility of the scene.
        Returns (is_valid, list_of_errors)
        """
        errors = []

        # Check for cycles in vertical relations
        vertical_rels = [Relation.ON, Relation.ON_TOP_OF, Relation.ABOVE,
                        Relation.UNDER, Relation.BELOW, Relation.BENEATH]

        # Build support graph
        supports = defaultdict(set)
        supported_by = defaultdict(set)

        for rel in self.relations:
            if rel.relation in [Relation.ON, Relation.ON_TOP_OF]:
                supports[rel.object.id].add(rel.subject.id)
                supported_by[rel.subject.id].add(rel.object.id)

        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in supports[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for obj_id in self.object_map:
            if obj_id not in visited:
                if has_cycle(obj_id, visited, set()):
                    errors.append(f"Cycle detected in support relations involving {obj_id}")

        # Check for contradictory relations
        for i, rel1 in enumerate(self.relations):
            for rel2 in self.relations[i+1:]:
                # Same objects but contradictory relations
                if rel1.subject == rel2.subject and rel1.object == rel2.object:
                    if (rel1.relation == Relation.LEFT_OF and rel2.relation == Relation.RIGHT_OF) or \
                       (rel1.relation == Relation.RIGHT_OF and rel2.relation == Relation.LEFT_OF):
                        errors.append(f"{rel1.subject.id} cannot be both left and right of {rel1.object.id}")

                    if (rel1.relation in [Relation.ON, Relation.ON_TOP_OF] and
                        rel2.relation in [Relation.UNDER, Relation.BELOW, Relation.BENEATH]):
                        errors.append(f"{rel1.subject.id} cannot be both on and under {rel1.object.id}")

        # Check stacking height (max 4 high)
        for obj_id in self.object_map:
            height = 0
            current = obj_id
            seen = set()
            while current in supported_by and supported_by[current]:
                if current in seen:
                    break  # Cycle, already caught above
                seen.add(current)
                current = list(supported_by[current])[0]
                height += 1
                if height > 4:
                    errors.append(f"Stacking height exceeds 4 for object {obj_id}")
                    break

        return len(errors) == 0, errors

    def to_canonical(self) -> str:
        """Return a canonical string representation for comparison"""
        # Sort objects by ID
        objs = sorted(self.objects, key=lambda o: o.id)
        # Sort relations
        rels = sorted(self.relations,
                     key=lambda r: (r.subject.id, r.relation.value, r.object.id))

        canonical = []
        for obj in objs:
            canonical.append(f"{obj.id}:{obj.color.value}_{obj.shape.value}")
        canonical.append("|")
        for rel in rels:
            canonical.append(f"{rel.subject.id}_{rel.relation.value}_{rel.object.id}")

        return ";".join(canonical)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations]
        }

    def get_scene_hash(self) -> str:
        """Get a unique hash for this scene configuration"""
        canonical = self.to_canonical()
        return hashlib.md5(canonical.encode()).hexdigest()[:8]


class SceneGenerator:
    """Generates random scenes with specified complexity"""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.object_counter = 0

        # Define which colors/shapes to use at each complexity level
        self.complexity_config = {
            1: {
                "num_objects": (2, 3),
                "num_relations": (1, 2),
                "colors": [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW],
                "shapes": [Shape.BLOCK, Shape.CUBE, Shape.BOX],
                "relations": [Relation.ON, Relation.UNDER, Relation.LEFT_OF,
                             Relation.RIGHT_OF, Relation.NEXT_TO]
            },
            2: {
                "num_objects": (3, 4),
                "num_relations": (2, 3),
                "colors": [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW,
                          Color.ORANGE, Color.PURPLE],
                "shapes": [Shape.BLOCK, Shape.CUBE, Shape.BOX],
                "relations": [Relation.ON, Relation.ON_TOP_OF, Relation.UNDER,
                             Relation.BELOW, Relation.LEFT_OF, Relation.RIGHT_OF,
                             Relation.NEXT_TO, Relation.BESIDE, Relation.NEAR]
            },
            3: {
                "num_objects": (4, 6),
                "num_relations": (3, 5),
                "colors": list(Color)[:10],  # First 10 colors
                "shapes": [Shape.BLOCK, Shape.CUBE, Shape.BOX, Shape.SPHERE],
                "relations": [r for r in Relation if r not in [Relation.IN, Relation.INSIDE, Relation.CONTAINS]]
            }
        }

    def generate_scene(self, complexity: int = 1) -> SceneGraph:
        """Generate a random scene with specified complexity"""
        config = self.complexity_config.get(complexity, self.complexity_config[1])

        # Determine number of objects and relations
        num_objects = random.randint(*config["num_objects"])
        num_relations = random.randint(*config["num_relations"])

        scene = SceneGraph()

        # Generate objects
        used_combinations = set()
        for i in range(num_objects):
            # Try to avoid duplicate color-shape combinations
            attempts = 0
            while attempts < 50:
                color = random.choice(config["colors"])
                shape = random.choice(config["shapes"])
                combo = (color, shape)

                if combo not in used_combinations or attempts > 30:
                    used_combinations.add(combo)
                    obj = ColoredBlock(
                        id=f"obj_{self.object_counter:03d}",
                        color=color,
                        shape=shape
                    )
                    scene.add_object(obj)
                    self.object_counter += 1
                    break
                attempts += 1

        # Generate relations
        attempts = 0
        while len(scene.relations) < num_relations and attempts < 100:
            attempts += 1

            # Pick two different objects
            if len(scene.objects) < 2:
                break

            subj, obj = random.sample(scene.objects, 2)
            rel_type = random.choice(config["relations"])

            # Check if this relation already exists
            existing = any(r.subject == subj and r.object == obj
                          for r in scene.relations)
            if existing:
                continue

            relation = SpatialRelation(subj, rel_type, obj)

            # Temporarily add and validate
            scene.add_relation(relation)
            is_valid, _ = scene.validate()

            if not is_valid:
                # Remove the invalid relation
                scene.relations.pop()

            # For vertical relations, consider adding the inverse
            if rel_type in [Relation.ON, Relation.UNDER] and random.random() < 0.3:
                inverse = relation.inverse()
                if inverse and not any(r.subject == inverse.subject and
                                      r.object == inverse.object
                                      for r in scene.relations):
                    scene.add_relation(inverse)

        return scene

    def generate_connected_scene(self, complexity: int = 1) -> SceneGraph:
        """
        Generate a scene where all objects are connected through relations.
        Better for training as it avoids isolated objects.
        """
        scene = self.generate_scene(complexity)

        # Ensure connectivity
        connected = set()
        if scene.relations:
            # Start with objects from first relation
            first_rel = scene.relations[0]
            connected.add(first_rel.subject)
            connected.add(first_rel.object)

            # Try to connect any unconnected objects
            unconnected = [obj for obj in scene.objects if obj not in connected]

            for obj in unconnected:
                # Try to relate to a connected object
                target = random.choice(list(connected))
                rel_type = random.choice(self.complexity_config[complexity]["relations"])

                if random.random() < 0.5:
                    relation = SpatialRelation(obj, rel_type, target)
                else:
                    relation = SpatialRelation(target, rel_type, obj)

                scene.add_relation(relation)
                connected.add(obj)

        # Validate and regenerate if needed
        is_valid, errors = scene.validate()
        if not is_valid and complexity == 1:
            # For simple scenes, just regenerate
            return self.generate_connected_scene(complexity)

        return scene


class TextRenderer:
    """Converts SceneGraph to natural language sentences"""

    def __init__(self):
        # Templates for different complexity levels
        self.simple_templates = [
            "{subj} is {rel} {obj}",
            "{subj} sits {rel} {obj}",
            "{subj} is placed {rel} {obj}",
            "there is {subj} {rel} {obj}",
            "{obj} has {subj} {rel_inv} it",
        ]

        self.relation_phrases = {
            Relation.ON: ["on", "on top of", "sitting on", "placed on", "resting on"],
            Relation.ON_TOP_OF: ["on top of", "atop", "above"],
            Relation.UNDER: ["under", "beneath", "below", "underneath"],
            Relation.BELOW: ["below", "beneath"],
            Relation.LEFT_OF: ["to the left of", "left of", "on the left side of"],
            Relation.RIGHT_OF: ["to the right of", "right of", "on the right side of"],
            Relation.NEXT_TO: ["next to", "beside", "adjacent to"],
            Relation.BESIDE: ["beside", "next to"],
            Relation.NEAR: ["near", "close to", "nearby"],
        }

        self.determiners = ["the", "a", "this", "that"]

    def render_simple(self, scene: SceneGraph,
                     use_variety: bool = False) -> List[str]:
        """
        Render scene to simple sentences.
        Returns multiple possible renderings.
        """
        if not scene.relations:
            return []

        sentences = []

        for relation in scene.relations:
            # Choose relation phrase
            if use_variety and relation.relation in self.relation_phrases:
                rel_phrase = random.choice(self.relation_phrases[relation.relation])
            else:
                rel_phrase = relation.relation.value.replace("_", " ")

            # Choose determiners
            det1 = random.choice(self.determiners) if use_variety else "the"
            det2 = random.choice(self.determiners) if use_variety else "the"

            # Build subject and object descriptions
            subj_text = f"{det1} {relation.subject.color.value} {relation.subject.shape.value}"
            obj_text = f"{det2} {relation.object.color.value} {relation.object.shape.value}"

            # Choose template
            template = random.choice(self.simple_templates) if use_variety else self.simple_templates[0]

            # Handle inverse relation in template
            if "{rel_inv}" in template:
                inverse = relation.inverse()
                if inverse and inverse.relation in self.relation_phrases:
                    rel_phrase_inv = random.choice(self.relation_phrases[inverse.relation]) if use_variety else inverse.relation.value.replace("_", " ")
                    sentence = template.format(
                        subj=subj_text,
                        obj=obj_text,
                        rel_inv=rel_phrase_inv
                    )
                    sentences.append(sentence)
            else:
                sentence = template.format(
                    subj=subj_text,
                    obj=obj_text,
                    rel=rel_phrase
                )
                sentences.append(sentence)

        return sentences

    def render_complex(self, scene: SceneGraph) -> List[str]:
        """
        Render scene with complex sentences (conjunctions, multiple relations).
        """
        if len(scene.relations) < 2:
            return self.render_simple(scene, use_variety=True)

        sentences = []

        # Conjunction of two relations
        if len(scene.relations) >= 2:
            rel1, rel2 = random.sample(scene.relations, 2)

            # Check if they share a subject
            if rel1.subject == rel2.subject:
                subj_text = f"the {rel1.subject.color.value} {rel1.subject.shape.value}"
                obj1_text = f"the {rel1.object.color.value} {rel1.object.shape.value}"
                obj2_text = f"the {rel2.object.color.value} {rel2.object.shape.value}"

                rel1_phrase = random.choice(self.relation_phrases.get(rel1.relation, [rel1.relation.value.replace("_", " ")]))
                rel2_phrase = random.choice(self.relation_phrases.get(rel2.relation, [rel2.relation.value.replace("_", " ")]))

                sentences.append(
                    f"{subj_text} is {rel1_phrase} {obj1_text} and {rel2_phrase} {obj2_text}"
                )
                sentences.append(
                    f"{subj_text} that is {rel1_phrase} {obj1_text} is also {rel2_phrase} {obj2_text}"
                )

        # Chain of relations
        if len(scene.relations) >= 2:
            # Try to find a chain
            for rel1 in scene.relations:
                for rel2 in scene.relations:
                    if rel1.object == rel2.subject and rel1 != rel2:
                        # Found a chain: rel1.subject -> rel1.object/rel2.subject -> rel2.object
                        subj_text = f"the {rel1.subject.color.value} {rel1.subject.shape.value}"
                        mid_text = f"the {rel1.object.color.value} {rel1.object.shape.value}"
                        obj_text = f"the {rel2.object.color.value} {rel2.object.shape.value}"

                        rel1_phrase = random.choice(self.relation_phrases.get(rel1.relation, [rel1.relation.value.replace("_", " ")]))
                        rel2_phrase = random.choice(self.relation_phrases.get(rel2.relation, [rel2.relation.value.replace("_", " ")]))

                        sentences.append(
                            f"{subj_text} is {rel1_phrase} {mid_text}, which is {rel2_phrase} {obj_text}"
                        )
                        break

        # Add simple sentences too
        sentences.extend(self.render_simple(scene, use_variety=True)[:2])

        return sentences


def main():
    """Test the scene generation and rendering"""
    generator = SceneGenerator(seed=42)
    renderer = TextRenderer()

    print("Generating sample scenes and sentences...\n")

    for complexity in [1, 2]:
        print(f"\n{'='*50}")
        print(f"Complexity Level {complexity}")
        print('='*50)

        for i in range(3):
            scene = generator.generate_connected_scene(complexity)

            print(f"\nScene {i+1}:")
            print(f"  Objects: {len(scene.objects)}")
            for obj in scene.objects:
                print(f"    - {obj.id}: {obj.color.value} {obj.shape.value}")

            print(f"  Relations: {len(scene.relations)}")
            for rel in scene.relations:
                print(f"    - {rel.subject.id} {rel.relation.value} {rel.object.id}")

            # Validate
            is_valid, errors = scene.validate()
            print(f"  Valid: {is_valid}")
            if errors:
                for error in errors:
                    print(f"    ERROR: {error}")

            # Render to text
            simple_sentences = renderer.render_simple(scene, use_variety=False)
            varied_sentences = renderer.render_simple(scene, use_variety=True)
            complex_sentences = renderer.render_complex(scene) if complexity > 1 else []

            print("  Simple sentences:")
            for s in simple_sentences[:3]:
                print(f"    - {s}")

            print("  Varied sentences:")
            for s in varied_sentences[:3]:
                print(f"    - {s}")

            if complex_sentences:
                print("  Complex sentences:")
                for s in complex_sentences[:2]:
                    print(f"    - {s}")

            # Export format
            print(f"  Scene hash: {scene.get_scene_hash()}")
            print(f"  Canonical: {scene.to_canonical()[:50]}...")


if __name__ == "__main__":
    main()