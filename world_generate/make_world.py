import xml.etree.ElementTree as ET
import os

def append_sdf_files(base_path: str, extra_path: str, output_path: str):
    # Load both SDF files
    base_tree = ET.parse(base_path)
    base_root = base_tree.getroot()
    base_world = base_root.find('world')

    extra_tree = ET.parse(extra_path)
    extra_root = extra_tree.getroot()
    extra_world = extra_root.find('world')

    # Append all children from extra_world to base_world
    for child in extra_world:
        base_world.append(child)

    # Write combined SDF to output
    base_tree.write(output_path, encoding='utf-8', xml_declaration=True)


import world_generation
world_generation.main()
append_sdf_files("training_base.sdf", "all_training.sdf", "all_training.sdf")


