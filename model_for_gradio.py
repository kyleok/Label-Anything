import cv2
from PIL import ImageFile
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import matplotlib.pyplot as plt
import numpy as np

checkpoints = {"vit_h": "sam_vit_h_4b8939.pth",
               "vit_b": "sam_vit_b_01ec64.pth",
               "vit_l": "sam_vit_l_0b3195.pth"}

device = "cuda"

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_masks(image, **kwargs):
    options = {
        'model_type': "vit_h",
        'param2': 'default_value2',
        'param3': 'default_value3'
    }
    options.update(kwargs)
    model_type = options['model_type']

    points_per_side = options['points_per_side']
    pred_iou_thresh = options['pred_iou_thresh']
    stability_score_thresh = options['stability_score_thresh']
    min_mask_region_area = options['min_mask_region_area']
    stability_score_offset = options['stability_score_offset']
    box_nms_thresh = options['box_nms_thresh']
    crop_n_layers = options['crop_n_layers']
    crop_nms_thresh = options['crop_nms_thresh']

    sam_checkpoint = checkpoints[model_type]
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # image = cv2.imread('asd.jpg')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(pred_iou_thresh)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area, stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh, crop_n_layers=crop_n_layers, crop_nms_thresh=crop_nms_thresh,
        output_mode='binary_mask')  # output_mode = 'binary_mask','uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
    masks = mask_generator.generate(image)
    return masks


def get_values_from_dicts_list(dicts_list, key):
    return [dictionary[key] for dictionary in dicts_list]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img


def SAM_inference(image, **kwargs):
    masks = get_masks(image, **kwargs)
    annotated_img = show_anns(masks)
    segmentations = get_values_from_dicts_list(masks, 'segmentation')
    bboxs = get_values_from_dicts_list(masks, 'bbox')
    torch.cuda.empty_cache()
    return segmentations, bboxs, annotated_img
