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


def convert_json_to_yolo(json_path: str, save_path: str = "../../data/newbbs/txt/"):
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

    file = f"0 "
    points = data["shapes"][0]['points']

    point1x = points[0][0]
    point1y = points[0][1]
    point2x = points[1][0]
    point2y = points[1][1]

    width = point2x - point1x
    height = point2y - point1y

    x_center = point1x+(width/2)
    y_center = point1y+(height/2)

    n_width = width/image_width
    n_height = height/image_height
    nx_center = x_center/image_width
    ny_center = y_center/image_height

    file += f"{nx_center} {ny_center} {n_width} {n_height} "

    # Save the YOLO formatted data to a text file
    save_path = save_path + "/" + json_path.split("/")[-1].replace(".json", ".txt")
    with open(save_path, 'w') as f:
        f.write(file)

if __name__ == "__main__":
    root_path = "../../data/newbbs"

    # Iterate through all JSON files in the directory
    for json_file in os.listdir(root_path):
        if json_file.endswith(".json"):
            json_path = os.path.join(root_path, json_file)
            convert_json_to_yolo(json_path)