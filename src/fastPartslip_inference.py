import sys
sys.path.insert(0,'../GroundingDINO')
sys.path.insert(0,'../SAM')
import os
import torch
import json
import numpy as np
from src.utils import normalize_pc,save_colored_pc,yolobbox2bbox,toDinoPrompt,check_pc_within_bbox
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

def InferDINO(category,xyz,pc_idx,screen_coords,TEXT_PROMPT,partList,modelDINO,predictorSAM,BOX_TRESHOLD = 0.2, TEXT_TRESHOLD = 0.3, SCORE_THRESHOLD=0.2,n_neighbors=21,n_pass=3, save_dir="tmp",experiment_name="Dino_KNN_semanticSeg"):
    if save_dir is not None:
        os.makedirs(f"{save_dir}/{experiment_name}/",exist_ok=True)
    preds = []
    for i in range(pc_idx.shape[0]):
        image_source, image = load_image(f"{save_dir}/rendered_img/{i}.png") #load rgb images
        predictorSAM.set_image(image_source)

        boxes, logits, phrases = predict(
                                        model=modelDINO,
                                        image=image,
                                        caption=TEXT_PROMPT,
                                        box_threshold=BOX_TRESHOLD,
                                        text_threshold=TEXT_TRESHOLD
                                    )
        phrases = np.array(phrases) #just to fix indexing

        xyxy = yolobbox2bbox(boxes)*image.shape[-1] #change bbox format to xyxy and scale with image size
        
        nms_mask = []
        for t,bbox in enumerate(xyxy): 
            if check_pc_within_bbox(bbox[0], bbox[1], bbox[2], bbox[3], screen_coords[i]).mean() < 0.95: 
                nms_mask.append(t)
                
        xyxy = xyxy[nms_mask]
        boxes = boxes[nms_mask]
        logits = logits[nms_mask]
        phrases = phrases[nms_mask]
        
        
        
        nms_indexes = nms(torch.tensor(xyxy) , logits, 0.5) #non maximum supression
        
        xyxy = xyxy[nms_indexes]
        boxes = boxes[nms_indexes]
        logits = logits[nms_indexes]
        phrases = phrases[nms_indexes]
        
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(f"{save_dir}/dino_pred_bbox/{i}.png", annotated_frame) #save an annotated image for DINO debugging
        
        
        final_indexes = []
        new_phrases = []
        for iindex,part in enumerate(phrases):
            for metaPart in partList:
                if part.find(metaPart) != -1:
                    final_indexes.append(iindex)
                    new_phrases.append(metaPart)
                    
        xyxy = xyxy[final_indexes]
        boxes = boxes[final_indexes]
        logits = logits[final_indexes]
        phrases = np.array(new_phrases)
       
        
        input_boxes = torch.tensor(xyxy, device=predictorSAM.device)    
        transformed_boxes = predictorSAM.transform.apply_boxes_torch(input_boxes, image_source.shape[:2])
        

        if transformed_boxes.shape[0] > 0:
            masks, _, _ = predictorSAM.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )    #create segmentation masks with sam

            for index,mask in enumerate(masks):
#                 print(mask[1:].shape)
                preds.append({'image_id': i, 'category_id':  phrases[index], 
                              'bbox': boxes[index]*image.shape[-1], 
                              'score': logits[index],
                              'mask':mask[0],
                              'image_source':image_source
                             }
                            )

    pc_aggMask = torch.zeros((xyz.shape[0],len(partList)+1)) #this is a segment agg mask we sum all the scores from our bboxes 
    #into their own respective channel, the lastpar channel is for unsegmented parts
    pc_aggMask[:,-1] = SCORE_THRESHOLD #we can set a confidence threshold by setting the unsegmented score
    for prediction in preds:
        maskedPC_idx = pc_idx[prediction["image_id"],prediction["mask"].cpu().numpy()] #this gives you the pc idx of the points that are inside the mask
        index_pcMasked = np.unique(maskedPC_idx)[1:] # we only need the unique idx and the first id is always -1 meaning not found
        pc_aggMask[index_pcMasked,partList[prediction["category_id"]]] += prediction["score"] #add up all the scores for each part
    pc_seg_classes = torch.argmax(pc_aggMask,dim=-1) #select the highest score as our segmentation class
    #if non of the part scores are over the SCORE_THRESHOLD it will be left unsegmented
    partColors = np.array(distinctipy.get_colors(len(partList)))
    accumulator = np.zeros((xyz.shape[0], len(partList)+1))
    accumulator[:,-1] = 1
    # since projections are not exact meaning not every PC point is rendered into our image our backprojections are not dense
    # use KNN to smooth these backprojections 
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(xyz) #create a knn

    for colorId,part in enumerate(partList):
        pc_part_idx = np.zeros((xyz.shape[0]),dtype=int)
        rgb_sem = np.zeros((xyz.shape[0],3))
        pc_part_idx[torch.where(pc_seg_classes==partList[part])] = 1
        
        for pass_ in range(n_pass):
            notColoredIndexes = torch.where(pc_seg_classes!=partList[part]) #find non segmented parts for smoothing

            n_indexes = nn.kneighbors(xyz[notColoredIndexes],n_neighbors+1,return_distance=False)
            n_indexes = n_indexes[:,1:] #get n_neighbors for the points, the first index is always the point itself so delete that
            #we have dense point clouds so distance based measures are not necessary and sometimes give worst results
            flag = pc_part_idx[n_indexes].mean(axis=1) 
            
            flag[np.where(flag>=0.4)] = 1 #and segmnent the points where the mean of neighbours are colored %40 or over
            flag[np.where(flag<0.4)] = 0
            pc_part_idx[notColoredIndexes] = flag
           
        rgb_sem[pc_part_idx.astype(bool)] = partColors[colorId]
        accumulator[pc_part_idx.astype(bool),colorId] += 1
        if save_dir is not None:
            save_colored_pc(f"{save_dir}/{experiment_name}/{part}.ply", xyz, rgb_sem)
        
    pc_partIDX = np.argmax(accumulator,axis=-1)
    partColors_extended = np.append(partColors,[[0,0,0]],axis=0)
    if save_dir is not None:
        save_colored_pc(f"{save_dir}/{experiment_name}/{category}.ply", xyz, partColors_extended[pc_partIDX.astype(int)]) 
        
    pc_partIDX[np.where(pc_partIDX == partColors_extended.shape[0]-1)] = -1
    return pc_partIDX