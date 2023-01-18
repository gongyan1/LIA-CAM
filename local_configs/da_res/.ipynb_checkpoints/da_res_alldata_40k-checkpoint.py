# ==========================================================================
# Model
# ==========================================================================
num_head_list = [1, 1, 1]
channels_list = [128, 256, 512]
# channels_list = [512]
backbone_stage_index = [1, 2, 3]
# backbone_stage_index = [3]
backbone_hw_list = [[60, 80], [30, 40], [15, 20]]
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size = (480, 640))
model = dict(
    type='Angle_EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    # HACK:Whether to automatically balance multiple losses, only for with_angle_loss
    # parse_loss=dict(type='ParseLoss', num_losses=2),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        style='pytorch'),
    # HACK:whether with angle fusion
    # neck=dict(
    #     type='Angle_Fusion_Neck',
    #     hw_list=backbone_hw_list,
    #     channels_list = channels_list,
    #     step_list=[8, 4, 2, 1],
    #     index_list=backbone_stage_index,
    #     num_head_list=num_head_list),
    decode_head=dict(
        type='Angle_DAHead',
        pam_channels=128,
        in_channels=channels_list,
        in_index=backbone_stage_index,
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        upsample_scale_factor_list=[2, 2, 2],
        # HACK:whether with angle loss
        #with_angle_loss=True,
        #num_pre_angle_layers=4,
        align_corners=False,
        input_transform='resize_concat',
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.4, 1])),
    test_cfg=dict(mode='whole'))


# ==========================================================================
# Dataset
# ==========================================================================
data_root = '/root/autodl-tmp/MyDataWithAngle_all'
crop_size = (480, 640)
dataset_type = 'Laneimg_Withangle'
train_pipeline = [
    dict(type='AngleLoadImageFromFile'),
    dict(type='AngleLoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(640, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='AnglePackSegInputs')
]
test_pipeline = [
    dict(type='AngleLoadImageFromFile'),
    dict(type='Resize', scale=(480, 640), keep_ratio=True),
    dict(type='AngleLoadAnnotations'),
    dict(type='AnglePackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='train.txt',
        data_root=data_root,
        data_prefix=dict(img_path='data/img', seg_map_path='data/label'),
        metainfo=dict(
            classes=('background', 'lane'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='val.txt',
        data_root=data_root,
        data_prefix=dict(img_path='data/img', seg_map_path='data/label'),
        metainfo=dict(
            classes=('background', 'lane'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='val.txt',
        data_root=data_root,
        data_prefix=dict(img_path='data/img', seg_map_path='data/label'),
        metainfo=dict(
            classes=('background', 'lane'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

# ==========================================================================
# Default runtime
# ==========================================================================
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', by_epoch=False)
log_level = 'INFO'
load_from = None
resume = None

# ==========================================================================
# Shedules
# ==========================================================================
randomness = dict(seed=42)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=600000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=600000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, show=False, interval=400))
launcher = 'none'