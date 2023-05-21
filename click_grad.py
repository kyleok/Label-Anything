import gradio as gr
from model_for_gradio import SAM_inference
import cv2
import numpy as np
import time
from distinctipy import distinctipy
import utils


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime of {func.__name__}: {runtime} seconds")
        return result

    return wrapper


with gr.Blocks(theme=gr.themes.Default(), title="Label Anything") as demo:
    masks = []
    selected_masks = dict()
    color_map = dict()


    def visualize_segmentation(image, transparency=0.7):
        global masks
        annotated_image = image.copy()

        for candidate in selected_masks:
            mask = masks[candidate]
            # Convert the binary mask to a numpy array
            mask_np = np.array(mask, dtype=np.uint8)

            # Create a mask with the same dimensions as the image for overlaying
            overlay_mask = np.zeros(image.shape, dtype=np.uint8)
            class_name = selected_masks[candidate]
            mask_color = color_map[class_name]
            mask_color = tuple([int(x * 255) for x in mask_color])
            overlay_mask[np.where((mask_np != 0))] = mask_color

            # Apply transparency only to the overlay mask
            overlay_mask = cv2.addWeighted(
                overlay_mask, transparency, np.zeros(image.shape, dtype=np.uint8), 1 - transparency, 0
            )

            # Overlay the segmented image on the annotated image using the mask
            annotated_image = cv2.addWeighted(
                annotated_image, 1, overlay_mask, 1, 0
            )

        return annotated_image


    def img_click(img, class_name, evt: gr.SelectData):
        global selected_masks
        real_candidates = []

        def find_array_with_most_true(candidates):
            maxi, smallest = np.inf, None
            for candidate in candidates:
                count = np.sum(masks[candidate])
                if count < maxi:
                    maxi, smallest = count, candidate
            return smallest

        click_coord = evt.index[0], evt.index[1]
        candidates = []
        for bbox in bboxes:
            x, y, w, h = bbox
            if x < click_coord[0] < x + w and y < click_coord[1] < y + h:
                candidates.append(bboxes.index(bbox))
        for candidate in candidates:
            if masks[candidate][click_coord[1]][click_coord[0]]:
                real_candidates.append(candidate)
        smallest = find_array_with_most_true(real_candidates)
        if smallest in selected_masks:
            del selected_masks[smallest]
        else:
            selected_masks[smallest] = class_name
        res = visualize_segmentation(img)
        return res


    @measure_runtime
    def run_inference(img_input, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                      stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh):
        global masks
        global bboxes
        global selected_masks
        global color_map
        selected_masks = dict()
        color_map = dict()
        res = SAM_inference(img_input, model_type="vit_h", points_per_side=points_per_side,
                            pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh,
                            min_mask_region_area=min_mask_region_area, stability_score_offset=stability_score_offset,
                            box_nms_thresh=box_nms_thresh, crop_n_layers=crop_n_layers, crop_nms_thresh=crop_nms_thresh)

        masks, bboxes, annotated_img = res
        print('inference done')
        return img_input, annotated_img


    def update_preview(img_input):
        return img_input


    def update_radio(category_names_string):
        global color_map
        category_names = category_names_string.split(',')
        # category_names.append('None')
        category_names = [x.strip() for x in category_names]
        colors = distinctipy.get_colors(len(category_names))
        print(colors)
        color_map = dict(zip(category_names, colors))
        print(color_map)
        return gr.update(choices=category_names, label="Class")


    def select_section(evt: gr.SelectData):
        return evt.value


    def select_section2(category_name):
        return category_name


    def json_prep(file_type='coco'):
        utils.masks_to_coco_polygon_json(masks, selected_masks)
        return "./coco_polygon.json"


    with gr.Column():
        gr.Markdown(
            """
            # Label Anything 🏷️
            Label Anything is a web-based application for object segmentation task using Segment Anything Model (SAM) by Meta AI.<br/>
            It helps user to reduce the labor of manual labeling for dataset building, as providing automatically generated regions of segmentation.<br/> 
            User can simply choose to which classes those segments belong.
            """)
        with gr.Accordion("Parameter Options", open=False):
            with gr.Row():
                points_per_side = gr.Number(value=32, label="points_per_side", interactive=True, precision=0,
                                            info='''The number of points to be sampled along one side of the image. The total 
                                            number of points is points_per_side**2.''')
                pred_iou_thresh = gr.Slider(value=0.88, minimum=0, maximum=1.0, step=0.01, label="pred_iou_thresh",
                                            interactive=True,
                                            info='''A filtering threshold in [0,1], using the model's predicted mask quality. The lower the value, the more masks will be generated.''')
                stability_score_thresh = gr.Slider(value=0.95, minimum=0, maximum=1.0, step=0.01,
                                                   label="stability_score_thresh", interactive=True,
                                                   info='''A filtering threshold in [0,1], using the stability of the mask under 
                                                   changes to the cutoff used to binarize the model's mask predictions.''')
                min_mask_region_area = gr.Number(value=0, label="min_mask_region_area", precision=0, interactive=True,
                                                 info='''If >0, postprocessing will be applied to remove disconnected regions 
                                                 and holes in masks with area smaller than min_mask_region_area.''')
            with gr.Row():
                stability_score_offset = gr.Number(value=1, label="stability_score_offset", interactive=True,
                                                   info='''The amount to shift the cutoff when calculated the stability score.''')
                box_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="box_nms_thresh",
                                           interactive=True,
                                           info='''The box IoU cutoff used by non-maximal ression to filter duplicate masks.''')
                crop_n_layers = gr.Number(value=0, label="crop_n_layers", precision=0, interactive=True,
                                          info='''If >0, mask prediction will be run again on crops of the image. 
                                          Sets the number of layers to run, where each layer has 2**i_layer number of image crops.''')
                crop_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="crop_nms_thresh",
                                            interactive=True,
                                            info='''The box IoU cutoff used by non-maximal suppression to filter duplicate 
                                            masks between different crops.''')

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="Original Image")
                segment_btn = gr.Button("Run Segmentation")

                category_names_string = gr.Textbox(
                    label="Categories (i.e., classes), comma-separated",
                    placeholder="E.g. car, bus, person",
                )
                class_update_btn = gr.Button("Update Class")
                class_radio = gr.Radio(choices=['None'], label="Class", interactive=True)

            with gr.Column(scale=2):
                img_show = gr.Image(label="Labeled Region")
                img_annot = gr.Image(label="Annotated Region")

            with gr.Column(scale=1):
                export_type = gr.Dropdown(["COCO", "others will be updated"], label="Export Format")
                #export_btn = gr.Button("Export!")
                out_download = gr.File(label="Output")
                jsonbtn = gr.Button("json download")

    class_update_btn.click(update_radio, [category_names_string], class_radio)

    segment_btn.click(run_inference,
                      [img_input, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                       stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh], [img_show, img_annot])
    jsonbtn.click(json_prep, export_type, out_download)
    img_show.select(img_click, [img_input, class_radio], img_show)

if __name__ == "__main__":
    demo.launch(share=True)
