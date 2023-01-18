import argparse
import torch
from mmcv.cnn import get_model_complexity_info
from mmengine import Config

from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules

register_all_modules(init_default_scope=True)

cfg = Config.fromfile('/root/angle-lane-mmseg/local_configs/da_res/da_res_mmlab_alldata.py')
cfg.model.pretrained = None
model = build_segmentor(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg')).cuda()
model.eval()

device = torch.device("cuda:0")
model.to(device)

image = torch.rand(1, 3, 480, 640).cuda()

for _ in range(50):  # GPU预热
    o = model(image)


starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
iterations = 300  # 重复计算的轮次
times = torch.zeros(iterations)  # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        o = model(image)
        ender.record()
        torch.cuda.synchronize()  # 同步GPU时间
        curr_time = starter.elapsed_time(ender)  # 计算时间
        times[iter] = curr_time

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))


# from thop import profile
# from thop import clever_format
# flops, params = profile(model, inputs=[image])
# flops, params = clever_format([flops, params], '%.3f')
# print('模型参数：', params)
# print('每一个样本浮点运算量：', flops)