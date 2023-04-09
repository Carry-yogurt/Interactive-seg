# import argparse
# import importlib.util
#
# import torch
# from isegm.utils.exp import init_experiment
# from models.sbd import r34_dh128
#
# def main():
#     args = parse_args()
#     torch.cuda.empty_cache()
#     model_script = load_module(args.model_path)
#     # 获取配置环境
#     cfg = init_experiment(args)
#     # 用于加速网络
#     torch.backends.cudnn.benchmark = True
#     sta = torch.multiprocessing.get_all_sharing_strategies()
#     torch.multiprocessing.set_sharing_strategy('file_system')
#     # 执行models.sbd目录下具体的网络
#     model_script.main(cfg)
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('model_path', type=str,
#                         help='Path to the model script.')
#
#     parser.add_argument('--exp-name', type=str, default='',
#                         help='Here you can specify the name of the experiment. '
#                              'It will be added as a suffix to the experiment folder.')
#
#     parser.add_argument('--workers', type=int, default=4,
#                         metavar='N', help='Dataloader threads.')
#
#     parser.add_argument('--batch-size', type=int, default=2,
#                         help='You can override model batch size by specify positive number.')
#
#     parser.add_argument('--ngpus', type=int, default=1,
#                         help='Number of GPUs. '
#                              'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
#                              'You should use either this argument or "--gpus".')
#
#     parser.add_argument('--gpus', type=str, default='', required=False,
#                         help='Ids of used GPUs. You should use either this argument or "--ngpus".')
#
#     parser.add_argument('--resume-exp', type=str, default=None,
#                         help='The prefix of the name of the experiment to be continued. '
#                              'If you use this field, you must specify the "--resume-prefix" argument.')
#
#     parser.add_argument('--resume-prefix', type=str, default='latest',
#                         help='The prefix of the name of the checkpoint to be loaded.')
#
#     parser.add_argument('--start-epoch', type=int, default=0,
#                         help='The number of the starting epoch from which training will continue. '
#                              '(it is important for correct logging and learning rate)')
#
#     parser.add_argument('--weights', type=str, default=None,
#                         help='Model weights will be loaded from the specified path if you use this argument.')
#
#     return parser.parse_args()
#
#
# def load_module(script_path):
#     spec = importlib.util.spec_from_file_location("model_script", script_path)
#     model_script = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(model_script)
#
#     return model_script
#

## 新的分支
import numpy as  np
from collections import OrderedDict
import torch.nn as nn

def params_count(model):
        """
        Compute the number of parameters.
        Args:
            model (model): model to count the number of parameters.
        """
        return np.sum([p.numel() for p in model.parameters()]).item()

from uint.isegm.utils.exp_imports.default import *
MODEL_NAME = 'hrnet18'



from thop import profile
from torch.utils.data import DataLoader
from uint.isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict

def test():
    model = DeeplabModel(backbone='resnet50', deeplab_ch=128, aspp_dropout=0.20, use_leaky_relu=True,
                  use_rgb_conv=True, use_disks=False, norm_radius=5)

    # model = SegFormerModel_b0_new_04_fpn()
    # model = HRNetModel2(width=18, ocr_width=64, with_aux_output=False, use_leaky_relu=True,
    #                    use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=False)

    # DeeplabMitb0_distMap      2971112166.0 4431793.0
    # DeeplabMobileNetv2_distMap  7550659780.0 3163985.0
    #Deeplabresnet18            41800570662.0 13242769.0
    #deeplaeff                  1194937015.0 612673.0
    # #SegFormer_effic_disk    3302284553.0 431689.0
    #SegFormer_mobile_disk  9203008054.0 2197329.0
    #SegFormer_resnet18_disk 10144107577.0 11695877.0
    #SegFormerModel_b0_new    4944015282.0 3712275.0

    # model  =UNet()
    import torch
    torch.save(model,"./fdasfda.pth")
    points_sampler = MultiPointSampler(20, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)
    trainset = SBDDataset(
        "./uint/data/sbd",
        split='train',
        augmentator=None,
        min_object_area=80,
        keep_background_prob=0.0,
        points_sampler=points_sampler,
        samples_scores_path='./assets/sbd_samples_weights.pkl',
        samples_scores_gamma=1.25
    )
    train_data = DataLoader(
        trainset, 1,
        sampler=get_sampler(trainset, shuffle=True, distributed=False),
        drop_last=True, pin_memory=True,
        num_workers=1
    )
    t = 0
    for i,d in enumerate(train_data):
        t=t+1
        if t ==10:
            flops, params = profile(model, inputs=(d['images'],d['points']))
            print(flops, params)
            break
# 100243963882.0 31396604.0
# 5621460402.0  3779571.0   segb0
# 5621460402.0  3779571.0    segb0
# 89933072772.0 31405841.0 deeplab
# 89933040772.0 31405841.0 deeplab
# 14909369686.0 14195411.0 segb1
# 15504315058.0 14195411.0 segb1
# 12065134216.0 3768211.0 segformer
# 64676230788.0 23350929.0
# 102680222979.0 31405841.0
# 156909252390.0 50397969.0

if __name__ == '__main__':
    # main()

    # model1 =   SegFormerModel_b0same_new_04_fpn()
    # model2 = DeeplabModel(backbone='resnet34', deeplab_ch=128, aspp_dropout=0.20, use_leaky_relu=True,
    #              use_rgb_conv=False, use_disks=True, norm_radius=5)
    # model3 = DeeplabModel(backbone='resnet50', deeplab_ch=128, aspp_dropout=0.20, use_leaky_relu=True,
    #              use_rgb_conv=False, use_disks=True, norm_radius=5)
    # model4 = DeeplabModel(backbone='resnet101', deeplab_ch=128, aspp_dropout=0.20, use_leaky_relu=True,
    #              use_rgb_conv=False, use_disks=True, norm_radius=5)
    # print(params_count(model1))
    # print(params_count(model2))
    # print(params_count(model3))
    # print(params_count(model4))
    test()