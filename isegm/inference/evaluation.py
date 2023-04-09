from time import time
from pathlib import Path
import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, oracle_eval=False, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        item = dataset[index]

        if oracle_eval:
            gt_mask = torch.tensor(sample['instances_mask'], dtype=torch.float32)
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            predictor.opt_functor.mask_loss.set_gt_mask(gt_mask)
        _, sample_ious, _ = evaluate_sample(index,item['images1'],item['images'], sample['instances_mask'], predictor, **kwargs)
        #_, sample_ious, _ = evaluate_sample(item['images'], sample['instances_mask'], predictor, **kwargs)
       # _, sample_ious, _ = evaluate_sample(item['images'], sample['instances_mask'], predictor, **kwargs)

        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

# 评估一个图像
import cv2

def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image

# @lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))




def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(255, 0, 0), radius=4):

    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)


        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result
def evaluate_sample(idd,images1,image_nd, instances_mask, predictor, max_iou_thr,
                    pred_thr=0.49, max_clicks=20,k=2,th=1):
    clicker = Clicker(gt_mask=instances_mask,k=k,th=th)
    pred_mask = np.zeros_like(instances_mask)
    ious_list = []
    save_path = Path('./experiments/predictions_vis')
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):

        sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        # image = image.transpose(0,2).transpose(0,1)
        #
        # image = image.cpu().numpy()
        image_with_mask = draw_with_blend_and_clicks(image, pred_probs>10, clicks_list=clicks_list,pos_color=(255, 0, 0),neg_color=(0, 0, 255),radius=7)
        ## 边缘检测
        prob_map[ pred_probs > pred_thr] = 100
        prob_map[pred_probs <= pred_thr] = 0
        prob_map = prob_map.astype(np.uint8)
        mask = prob_map.copy()
        mask[:,:,0] = 255
        mask[:, :, 1] = 215
        mask[:, :, 2] = 100
        mask[prob_map == 0] = 0

        # prob_map = np.expand_dims(prob_map,axis=0)
        cv2.addWeighted(mask, 0.7, image_with_mask, 0.9, 0, image_with_mask)
        image_with_mask = draw_with_blend_and_clicks(image_with_mask, pred_probs>10, clicks_list=clicks_list,pos_color=(255, 0, 0),neg_color=(0, 0, 255),radius=7)

        contours, hierarchy = cv2.findContours(prob_map[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_with_mask, contours, -1, (0, 255, 0), 1)

        cv2.imwrite(str(sample_path), image_with_mask[:,:,[2,1,0]])




    with torch.no_grad():
        predictor.set_input_image(image_nd)

        for click_number in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr
            callback(images1, instances_mask, pred_probs, idd, click_number, clicker.clicks_list)
            iou = utils.get_iou(instances_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
