batch_size = 1
data_root = 'Lane_Withangle_test'

dataset_type = 'Laneimg_Withangle'
train_pipeline = [
    dict(type='MyLoadImageFromFile'),
    dict(type='MyLoadAnnotations'),
    dict(type='My_with_Angle_PackSegInputs')
]
test_pipeline = [
    dict(type='MyLoadImageFromFile'),
    dict(type='MyLoadAnnotations'),
    dict(type='My_with_Angle_PackSegInputs')
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='Laneimg_Withangle',
        ann_file='train.txt',
        data_root=data_root,
        data_prefix=dict(img_path='data/img', seg_map_path='data/label'),
        metainfo=dict(
            classes=('background', 'lane'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Laneimg_Withangle',
        ann_file='val.txt',
        data_root=data_root,
        data_prefix=dict(img_path='data/img', seg_map_path='data/label'),
        metainfo=dict(
            classes=('background', 'lane'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Laneimg_Withangle',
        ann_file='val.txt',
        data_root=data_root,
        data_prefix=dict(img_path='data/img', seg_map_path='data/label'),
        metainfo=dict(
            classes=('background', 'lane'),
            palette=[[0, 0, 0], [255, 255, 255]]),
        pipeline=test_pipeline))
