import sys
sys.path.insert(0,'../GroundingDINO')
sys.path.insert(0,'../SAM')
import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc,save_colored_pc,yolobbox2bbox,toDinoPrompt,check_pc_within_bbox
from src.render_pc import render_pc
from src.fastPartslip_inference import InferDINO
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from torchvision.ops import nms
import json
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import distinctipy
from glob import glob
from tqdm import tqdm
import time
from functools import wraps
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    metaData = json.load(open("./PartSlip/PartNetE_meta.json")) 
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    modelDINO = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                       "./GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
                      device=device
                      )

    sam_checkpoint = "./SAM/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")

    predictorSAM = SamPredictor(sam)
    
    io = IO()
    for category in ["Chair", "Suitcase", "Refrigerator", "Lamp", "Kettle"]: 
        input_pc_file = f"examples/{category}.ply"
        save_dir = f"examples/{category}_ZeroShot"
        xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
        img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_dir, device)
        TEXT_PROMPT,partList = toDinoPrompt(metaData,category)
        InferDINO(category,xyz,pc_idx,screen_coords,TEXT_PROMPT,partList,modelDINO,predictorSAM,
            BOX_TRESHOLD = 0.2, TEXT_TRESHOLD = 0.3, 
            SCORE_THRESHOLD=0.2,n_neighbors=21,n_pass=3, 
            save_dir=save_dir,experiment_name="semanticSeg")