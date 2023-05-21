import numpy as np
import json
import cv2


def export_binary_json(masks: list, labels: dict) -> dict:
    masks = [x.tolist() for x in masks]
    export_json = dict()

    # Pre define key & value pairs of final .json file
    shapes = []
    imagePath = ""
    imageHeight = 0
    imageWidth = 0

    # Get shape for each label (i.e., mask)
    for index in labels:
        shape = dict()

        mask = masks[index]
        shape['mask'] = mask  # Originally, this key would be "points"
        group_id = None
        shape_type = "polygon"

        shapes.append(shape)

    export_json['shapes'] = shapes
    export_json['imagePath'] = imagePath
    export_json['imageHeight'] = imageHeight
    export_json['imageWidth'] = imageWidth

    return export_json


def masks_to_coco_polygon_json(masks, category_mapping):
    annotations = []
    categories = list(set(category_mapping.values()))
    category_id_map = {y: x + 1 for x, y in enumerate(categories)}
    count = 0
    for idx, category in category_mapping.items():
        count += 1
        mask = masks[idx]
        polygons = []
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                polygons.append(contour)

        annotation = {
            "segmentation": [polygons],
            "category_id": category_id_map[category_mapping[idx]],
            "id": count
        }
        annotations.append(annotation)

    coco_data = {
        "annotations": annotations,
        "images": [
            {
                "file_name": 'image.jpg',
                "id": 1
            }
        ],
        "categories": [
            {
                "id": category_id_map[category],
                "name": category
            } for category in categories
        ]
    }
    output_file_name = "coco_polygon.json"
    with open(output_file_name, "w") as file:
        json.dump(coco_data, file)


if __name__ == "__main__":
    # Example usage
    masks = [
        np.array([
            [False, True, True, False],
            [True, True, False, True],
            [False, False, True, False]
        ]),
        np.array([
            [True, False, False, True],
            [False, True, True, False],
            [True, True, False, True]
        ]),
        np.array([
            [True, True, False, False],
            [False, True, True, True],
            [True, False, False, True]
        ]),
        np.array([
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False]
        ]),
        np.array([
            [True, True, True, False],
            [True, False, True, True],
            [False, True, False, True]
        ])
    ]
    category_mapping = {
        0: "object1",
        1: "object2",
        3: "object2"
    }

    masks_to_coco_polygon_json(masks, category_mapping)
