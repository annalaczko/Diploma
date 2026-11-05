"""Standalone one-off script: convert LabelMe annotations (one JSON file per
image) into YOLO bounding-box label files (one .txt per image), e.g. after
annotating a new batch of images by hand.

Not part of the Step/Pipeline classes; run it directly as a script.
"""

import json
import os


def convert_json_to_yolo(json_path: str, save_path: str = "../../data/newbbs/txt/"):
    """Convert one LabelMe JSON annotation into a YOLO detection label.

    The first shape's first two points are taken as opposite corners of
    the box. Writes `<save_path>/<same name>.txt` with one line,
    `0 x_center y_center width height`, all divided by the image's
    width/height (YOLO's normalized coordinates); 0 is the only class.
    Further shapes in the JSON are ignored.

    Args:
        json_path: one LabelMe JSON annotation file.
        save_path: folder the YOLO .txt label is written into.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_height = data["imageHeight"]
    image_width = data["imageWidth"]

    (point1x, point1y), (point2x, point2y) = data["shapes"][0]['points'][:2]
    width = point2x - point1x
    height = point2y - point1y
    x_center = point1x + (width / 2)
    y_center = point1y + (height / 2)

    line = (f"0 {x_center / image_width} {y_center / image_height} "
            f"{width / image_width} {height / image_height} ")

    save_path = save_path + "/" + json_path.split("/")[-1].replace(".json", ".txt")
    with open(save_path, 'w') as f:
        f.write(line)


if __name__ == "__main__":
    root_path = "../../data/newbbs"
    for json_file in os.listdir(root_path):
        if json_file.endswith(".json"):
            convert_json_to_yolo(os.path.join(root_path, json_file))
