import sys
sys.path.insert(0,'../GroundingDINO')
sys.path.insert(0,'../SAM')
import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc,save_colored_pc
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint
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

class ShapeNetParts(torch.utils.data.Dataset):
    def __init__(self, device, metadata):
        self.device = device 
        # Filtering is only to get quick results for a chair
        self.items = glob(f"./data/*/*")
        self.io = IO()
        self.metaData = metadata
    def __getitem__(self, index):
        path = self.items[index]

        category = path.split("/")[-2]
        code = path.split("/")[-1]
        
        segmentation_labels = np.load(f"{path}/label.npy",allow_pickle=True).item()
        xyz, rgb = normalize_pc(pc_file=f"{path}/pc.ply", save_dir="",io=self.io, device=self.device)
        pcidx_screencoords = np.load(f"{path}/pcidx_screencoords.npz")
        TEXT_PROMPT,partList = toDinoPrompt(self.metaData,category)
        
        return {
            'category': category,
            'xyz': xyz,
            'rgb': rgb,
            'pc_idx' : pcidx_screencoords["pc_idx"],
            'screen_coords' : pcidx_screencoords["screen_coords"],
            'TEXT_PROMPT' : TEXT_PROMPT,
            'partList':partList,
            'segmentation_labels': segmentation_labels["semantic_seg"],
            'save_dir': f"{path}",
            'code': f"{code}"
        }
    def __len__(self):
        return len(self.items)