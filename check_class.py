import xml.etree.ElementTree as ET
from pathlib import Path

# Change this path to your annotations folder
ann_folder = Path("car_plate_raw/annotations")

# Get the first XML file
first_xml = list(ann_folder.glob("*.xml"))[0]
print("Checking:", first_xml.name)

tree = ET.parse(first_xml)
root = tree.getroot()

# Find all object names
for obj in root.findall(".//object"):
    name_elem = obj.find(".//name")
    if name_elem is not None:
        print("Found object name:", repr(name_elem.text))
    else:
        print("No <name> tag found")