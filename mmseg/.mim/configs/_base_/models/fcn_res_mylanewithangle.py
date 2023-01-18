# model settings
backbone_stage_index = [1, 2, 3]
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='Mylane_with_angle_EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    pretrained=None,
    parse_loss=dict(type='ParseLoss', num_losses=2),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='Angle_Fusion_Neck',
        hw_list=[[60, 80], [30, 40], [15, 20]],
        step_list=[8, 4, 2, 1],
        index_list=[1, 2, 3]),
    decode_head=dict(
        type='My_Decode_Head',
        in_channels=[512, 1024, 2048],
        in_index=[1, 2, 3],
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        upsample_scale_factor_list=[2, 2, 2],
        with_angle_loss=True,
        num_pre_angle_layers=4,
        align_corners=False,
        input_transform='resize_concat',
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # mode = 'slide' Not tested
    test_cfg=dict(mode='whole'))