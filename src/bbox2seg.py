import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import save_colored_pc, get_iou

def check_pc_within_bbox(x1, y1, x2, y2, pc):  
    flag = np.logical_and(pc[:, 0] > x1, pc[:, 0] < x2)
    flag = np.logical_and(flag, pc[:, 1] > y1)
    flag = np.logical_and(flag, pc[:, 1] < y2)
    return flag

def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

def get_union(f, x): # union-find
    if f[x] == x:
        return x
    f[x] = get_union(f, f[x])
    return f[x]

def calc_sp_connectivity(xyz, superpoints, thr=0.02): 
# calculate connectivity (bounding box adjacency) between superpoints
    n = len(superpoints)
    X_min, X_max = [], []
    for i in range(n):
        X_min.append(xyz[superpoints[i], :].min(axis=0))
        X_max.append(xyz[superpoints[i], :].max(axis=0))
    X_min = np.array(X_min)
    X_max = np.array(X_max)
    A = (X_min.reshape(n, 1, 3) - X_max.reshape(1, n, 3)).max(axis=2)
    A = np.maximum(A, A.transpose())
    connectivity = A < thr
    return connectivity

def bbox2seg(xyz, superpoint, preds, screen_coor_all, point_idx_all, part_names, save_dir,
            num_view=10, solve_instance_seg=True, visualize=True):
#     print("semantic segmentation...")
    n_category = len(part_names)
    n_sp = len(superpoint)
    sp_visible_cnt = np.zeros(n_sp) #visible points for each superpoint
    sp_bbox_visible_cnt = np.zeros((n_category, n_sp)) 
    #visible points of superpoint j that are covered by at least a bounding box of category i
    preds_per_view = [[] for i in range(num_view)]
    for pred in preds:
        preds_per_view[pred["image_id"]].append(pred)
    in_box_ratio_list = [[[] for j in range(n_sp)] for i in range(n_category)] #used for instance segmentation
    visible_pts_list = []
    for i in range(num_view):
        screen_coor = screen_coor_all[i] #2D projected location of each 3D point
        point_idx = point_idx_all[i] #point index of each 2D pixel
        visible_pts = np.unique(point_idx)[1:] # the first one is -1
        visible_pts_list.append(visible_pts)
        valid_preds = []
        for pred in preds_per_view[i]:
            x1, y1, w, h = pred["bbox"]
            if check_pc_within_bbox(x1, y1, x1 + w, y1 + h, screen_coor).mean() < 0.98: 
                #ignore bbox covering the whole objects
                valid_preds.append(pred)
        for k, sp in enumerate(superpoint):
            sp_visible_pts = intersection(sp, visible_pts)
            sp_visible_cnt[k] += len(sp_visible_pts) 
            in_bbox = np.zeros((n_category, len(sp_visible_pts)), dtype=bool)
            if len(sp_visible_pts) != 0:
                sp_coor = screen_coor[sp_visible_pts]
                bb1 = {'x1': sp_coor[:, 0].min(), 'y1': sp_coor[:, 1].min(), \
                        'x2': sp_coor[:, 0].max(), 'y2': sp_coor[:, 1].max()}
            for pred in valid_preds:
                cat_id = pred["category_id"] - 1
                x1, y1, w, h = pred["bbox"]
                x2, y2 = x1 + w, y1 + h
                if len(sp_visible_pts) == 0:
                    in_box_ratio_list[cat_id][k].append(-1)
                else:
                    bb2 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    if get_iou(bb1, bb2) < 1e-6:
                        in_box_ratio_list[cat_id][k].append(0)
                    else:
                        mask = check_pc_within_bbox(x1, y1, x2, y2, sp_coor)
                        in_bbox[cat_id] = np.logical_or(in_bbox[cat_id], mask)
                        in_box_ratio_list[cat_id][k].append(mask.mean())
            for j in range(n_category):
                sp_bbox_visible_cnt[j, k] += in_bbox[j].sum() 

    sem_score = sp_bbox_visible_cnt / (sp_visible_cnt.reshape(1, -1) + 1e-6)
    sem_score[:, sp_visible_cnt == 0] = 0
    sem_seg = np.ones(xyz.shape[0], dtype=np.int32) * -1

    # assign semantic labels to superpoints
    for i in range(n_sp):
        if sem_score[:, i].max() < 0.5:
            continue
        idx = -1
        for j in reversed(range(n_category)): #give priority to small parts
            if sem_score[j, i] >= 0.5 and part_names[j] in ["handle", "button", "wheel", "knob", "switch", "bulb", "shaft", "touchpad", "camera", "screw"]:
                idx = j
                break
        if idx == -1:
            idx = np.argmax(sem_score[:, i])
        sem_seg[superpoint[i]] = idx
    if visualize:
        os.makedirs("%s/semantic_seg" % save_dir, exist_ok=True)  
        for j in range(n_category):
            rgb_sem = np.ones((xyz.shape[0], 3)) * (sem_seg == j).reshape(-1, 1)
            save_colored_pc("%s/semantic_seg/%s.ply" % (save_dir, part_names[j]), xyz, rgb_sem)
    if solve_instance_seg == False:
        return sem_seg, None
    
    print("instance segmentation...")
    os.makedirs("%s/instance_seg" % save_dir, exist_ok=True)
    connectivity = calc_sp_connectivity(xyz, superpoint)
    ins_seg = np.ones(xyz.shape[0], dtype=np.int32) * -1
    ins_cnt = 0
    for j in range(n_category):
        f = []
        for i in range(n_sp): # initialize union-find sets
            f.append(i)
        # merge superpoints that are adjacent and have similar bounding box ratio
        for i in range(n_sp):
            if sem_seg[superpoint[i][0]] == j:
                for k in range(i):
                    if sem_seg[superpoint[k][0]] == j and connectivity[i][k]:
                        ratio_i = np.array(in_box_ratio_list[j][i])
                        ratio_k = np.array(in_box_ratio_list[j][k])
                        mask = np.logical_and(ratio_i > -1, ratio_k > -1) # -1 indicates invisible
                        if mask.sum() == 0 or max(ratio_i[mask].sum(), ratio_k[mask].sum()) < 1e-3: 
                            dis = 1
                        else:
                            dis = np.abs(ratio_i[mask] - ratio_k[mask]).sum()
                            dis /= max(ratio_i[mask].sum(), ratio_k[mask].sum())
                        l1 = len(superpoint[i])
                        l2 = len(superpoint[k])
                        if dis < 0.2 and max(l1, l2) / min(l1, l2) < 100: 
                            f[get_union(f, i)] = get_union(f, k) # merge two union-find sets
        instances = []
        flags = []
        merged_sps = [[] for i in range(n_sp)]
        for i in range(n_sp):
            merged_sps[get_union(f, i)].append(superpoint[i])
        for i in range(n_sp):
            if len(merged_sps[i]) > 0 and sem_seg[superpoint[i][0]] == j:
                instances.append(np.concatenate(merged_sps[i]))
                flags.append(False)

        #filter out instances that have small iou with all bounding boxes
        for i in range(num_view):
            screen_coor = screen_coor_all[i] #2D projected location of each 3D point
            visible_pts = visible_pts_list[i]
            for k, instance in enumerate(instances):
                if flags[k]:
                    continue
                ins_visible_pts = intersection(instance, visible_pts)
                if len(ins_visible_pts) == 0:
                    continue
                ins_coor = screen_coor[ins_visible_pts]
                bb1 = {'x1': ins_coor[:, 0].min(), 'y1': ins_coor[:, 1].min(), \
                        'x2': ins_coor[:, 0].max(), 'y2': ins_coor[:, 1].max()}
                for pred in preds_per_view[i]:
                    cat_id = pred["category_id"] - 1
                    if cat_id != j:
                        continue
                    x1, y1, w, h = pred["bbox"]
                    bb2 = {'x1': x1, 'y1': y1, 'x2': x1 + w, 'y2': y1 + h}
                    if get_iou(bb1, bb2) > 0.5:
                        flags[k] = True
                        break
        rgb_ins = np.zeros((xyz.shape[0], 3)) 
        for i in range(len(instances)):
            if flags[i]:
                ins_seg[instances[i]] = ins_cnt
                ins_cnt += 1
                rgb_ins[instances[i]] = np.random.rand(3)  
        if visualize:
            save_colored_pc("%s/instance_seg/%s.ply" % (save_dir, part_names[j]), xyz, rgb_ins)
    return sem_seg, ins_seg