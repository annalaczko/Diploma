"""Standalone one-off script: convert LabelMe-style polygon annotations
(one JSON file per image, as exported by the LabelMe tool) into YOLO
segmentation label files (one .txt per image, normalized polygon
coordinates).

Not part of the Step/Pipeline class hierarchy - run directly as a script
when you have a folder of LabelMe JSON exports to convert once, e.g. after
manually annotating a new batch of images.
"""

import json
import os


def convert_json_to_yolo(json_path: str, save_path: str = "../../data/yolo_data"):
    """Convert one LabelMe JSON annotation into a YOLO segmentation label.

    Writes `<save_path>/<same filename>.txt` with a single line:
    `1 x1 y1 x2 y2 ...` where each (x, y) is the polygon's point,
    normalized to [0, 1] by the image's width/height, and `1` is a
    hardcoded class id (this dataset only ever labels one class).

    Limitations (fine for this dataset, worth knowing if reused
    elsewhere): only the *first* shape in the JSON (`shapes[0]`) is
    converted - annotations with multiple polygons per image would lose
    everything past the first one.

    Args:
        json_path: path to a single LabelMe JSON annotation file.
        save_path: directory the YOLO .txt label is written into.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_height = data["imageHeight"]
    image_width = data["imageWidth"]

    file = f"1 "
    points = data["shapes"][0]['points']

    for point in points:
        x = point[0]
        y = point[1]
        normalized_x = x / image_width
        normalized_y = y / image_height
        file += f"{normalized_x} {normalized_y} "

    # Save the YOLO formatted data to a text file
    save_path = save_path + "/" + json_path.split("/")[-1].replace(".json", ".txt")
    with open(save_path, 'w') as f:
        f.write(file)

if __name__ == "__main__":
    root_path = "../../data/interim/test/json"

    # Iterate through all JSON files in the directory
    for json_file in os.listdir(root_path):
        if json_file.endswith(".json"):
            json_path = os.path.join(root_path, json_file)
            convert_json_to_yolo(json_path)