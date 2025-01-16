from demoDINO import InferDINOSAMZeroShot
from demo import InferGLIP

from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model
from src.glip_inference import load_model_glip

from src.utils import normalize_pc,save_colored_pc
from src.render_pc import render_pc

import numpy as np
import json
import torch
from pytorch3d.io import IO
import os

def get_files_from_txt(file_path, categories, num_files):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    parsed_data = [(int(line.split()[0]), line.split()[1]) for line in lines]
    
    results = {}
    for category in categories:
        filtered_data = [f"data/{category}/{item[0]}" for item in parsed_data if item[1] == category]
        results[category] = filtered_data[:num_files]
    
    return results

def toDinoPrompt(metaData,className):
    listOfParts = metaData[className]
    prompt = ""
    partList = {}
    for i,part in enumerate(listOfParts):
        prompt += f"{className} {part}.".lower()
        partList[f"{className} {part}".lower()] = i
    return prompt,partList


def mIOU_f(label, predictions, case='dino', verbose=True):
    """
    Calculate mean IoU for semantic segmentation predictions.

    Args:
        label: Ground truth label containing 'semantic_seg'.
        predictions: Predictions dictionary from the model.
        case: Specify 'dino' or 'glip' for the respective models.
        verbose: If True, prints detailed IoU calculation logs.

    Returns:
        Tuple of (list of IoUs, mean IoU).
    """
    def log(msg):
        if verbose:
            print(msg)

    try:
        # Validate ground truth (label)
        if 'semantic_seg' not in label.item():
            raise ValueError("The label does not contain 'semantic_seg'.")
        semantic_seg = np.array(label.item()['semantic_seg'])
        unique_parts = np.unique(semantic_seg)
        unique_parts = unique_parts[unique_parts != -1]  # Exclude background (-1)

        ious = []

        if case == 'dino':
            # Validate predictions for 'dino' case
            if 'partList' not in predictions or 'partseg_rgbs' not in predictions:
                raise ValueError(
                    "Invalid 'predictions' format for 'dino'. Expected keys: 'partList' and 'partseg_rgbs'."
                )
            part_list = predictions['partList']
            partseg_rgbs = predictions['partseg_rgbs']

            for part, part_id in part_list.items():
                if part_id not in unique_parts:
                    log(f"{part} not present in ground truth, skipping IoU calculation.")
                    continue

                gt_mask = semantic_seg == part_id
                pred_mask = np.any(partseg_rgbs[part] != [0., 0., 0.], axis=-1)

                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()

                iou = intersection / union if union > 0 else 0
                ious.append(iou)
                log(f"IoU for {part}: {iou:.4f}")

        elif case == 'glip':
            # Validate predictions for 'other' case
            if 'part_names_ordered' not in predictions or 'sem_seg' not in predictions:
                raise ValueError(
                    "Invalid 'predictions' format for 'glip'. Expected keys: 'part_names_ordered' and 'sem_seg'."
                )
            part_names = predictions['part_names_ordered']
            sem_seg_pred = np.array(predictions['sem_seg'])
            category = str(predictions['category'])
            
            for idx, part_name in enumerate(part_names):
                if idx not in unique_parts:
                    log(f"{category.lower()} {part_name} not present in ground truth, skipping IoU calculation.")
                    continue

                gt_mask = semantic_seg == idx
                pred_mask = sem_seg_pred == idx

                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()

                iou = intersection / union if union > 0 else 0
                ious.append(iou)
                log(f"IoU for {part_name}: {iou:.4f}")

        else:
            # Raise an error for unsupported cases
            raise ValueError(f"Unsupported case: {case}. Use 'dino' or 'glip'.")

        # Calculate mean IoU
        if ious:
            mean_iou = np.mean(ious)
            log(f"Mean IoU: {mean_iou:.4f}")
        else:
            log("No valid parts found in the ground truth. Mean IoU cannot be calculated.")
            mean_iou = None

        return ious, mean_iou

    except Exception as e:
        raise RuntimeError(f"Error calculating IoU: {e}") from e


if __name__ == "__main__":
    metaData = json.load(open("./PartNetE_meta.json")) 
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        
    modelDINO = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "../GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
                      device=device
                      )
    
    
    config ="GLIP/configs/glip_Swin_L.yaml"
    weight_path = "models/glip_large_model.pth"
    #     print("[loading GLIP model...]")
    glip_demo = load_model_glip(config, weight_path)
    
    sam_checkpoint = "../SAM/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    predictorSAM = SamPredictor(sam)
    
    categories = ["Table", "Switch", "Toilet", "USB", "Scissors", "Printer", "Pen", "Lighter", "Suitcase", "Refrigerator", "Lamp", "Kettle"]
    
    files = get_files_from_txt("test_files.txt", categories, 10)
    for category, file_items in files.items():
        print(category +" inference DINO, GLIP ==> SAM started" +":")
        for file in file_items:
            #     print("-----Zero-shot inference of %s-----" % input_pc_file)
            print("------Rendering------")
            TEXT_PROMPT,partList = toDinoPrompt(metaData, category)
            
            save_dir = f'examples/zeroshot_{category}'
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(f"{save_dir}/rendered_img", exist_ok=True) #create the necessary save directories
            os.makedirs(f"{save_dir}/dino_pred", exist_ok=True)
            os.makedirs(f"{save_dir}/semantic_segDino_KNN", exist_ok=True)

            io = IO()
            xyz, rgb = normalize_pc(file + "/pc.ply", save_dir, io, device) #read Point cloud and rgb in the format n,3
            img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_dir, device, file)
            print("------Rendering Completed------")
            print("----DINO Inference Starting----")
#             preds = InferDINOSAMZeroShot(TEXT_PROMPT, partList, xyz, pc_idx, screen_coords, category, 
#                                      modelDINO, predictorSAM, device, 
#                                      BOX_TRESHOLD=0.2, TEXT_TRESHOLD=0.3, 
#                                      SCORE_THRESHOLD=0.2, n_neighbors=21, 
#                                      n_pass=5, save_dir=save_dir)
#             print("----DINO Inference Completed---")
            
#             print("mIOU CALCULATION DINO, " + category + ":")
            label = np.load(file + "/label.npy", allow_pickle=True)
#             mIOU_list,mean_iou = mIOU_f(label, preds)
            
            print("----GLIP Inference Starting----")
            preds_glip = InferGLIP(xyz, rgb, screen_coords, pc_idx, category, metaData[category], glip_demo, device, file)
            print("----GLIP Inference Completed---")
            
            print("mIOU CALCULATION GLIP, " + category + ":")
            mIOU_list_glip, mean_iou_glip = mIOU_f(label, preds_glip, "glip")
            
        