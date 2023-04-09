from collections import namedtuple
import torch
import numpy as np
from copy import deepcopy
from scipy.ndimage import distance_transform_edt

Click = namedtuple('Click', ['is_positive', 'coords'])


class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1,k=2,th=1):
        self.k =k
        self.th = th
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = distance_transform_edt(fn_mask)
        fp_mask_dt = distance_transform_edt(fp_mask)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map
        k = self.k
        tempfn, _ = torch.tensor(fn_mask_dt).flatten().topk(k)

        tempfp, _ = torch.tensor(fp_mask_dt).flatten().topk(k)

        fn_max_dist = tempfn[k - 1].numpy()
        fp_max_dist = tempfp[k - 1].numpy()
        # 230 210
        if (fn_max_dist <= 0):
            fn_max_dist = np.max(fn_mask_dt)
        if (fp_max_dist <= 0):
            fp_max_dist = np.max(fp_mask_dt)
        ## 随机扰动
        if abs(fn_max_dist - fp_max_dist) < self.th:
            # is_positive = fn_max_dist > fp_max_dist
            # is_positive = fn_max_dist > fp_max_dist
            randomInt = np.random.randint(0, 2)
            if randomInt == 0 or fp_max_dist <= 1.0:
                is_positive = True
            elif randomInt == 1 or fn_max_dist <= 1.0:
                is_positive = False
        else:
            # 正常
            is_positive = fn_max_dist > fp_max_dist


        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt >= fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt >= fp_max_dist)  # coords is [y, x]
        randomindex = np.random.randint(0, coords_y.shape)[0]
        return Click(is_positive=is_positive, coords=(coords_y[randomindex], coords_x[randomindex]))

    def add_click(self, click):
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)
