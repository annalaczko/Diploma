import json
import os


def convert_json_to_yolo(json_path: str, save_path: str = "../../data/yolo_data" ):
        with open(json_path, 'r') as f:
            data = json.load(f)
        image_height = data["imageHeight"]
        image_width = data["imageWidth"]

        file = f"1 "
        points = data["shapes"][0]['points']

        for point in points:
            x=point[0]
            y=point[1]
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