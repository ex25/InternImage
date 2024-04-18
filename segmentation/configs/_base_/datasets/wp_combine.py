# dataset settings
dataset_type = 'WeatherProofDataset'
data_root = '/root/autodl-tmp/project/InternImage/segmentation/data/WeatherProof/WeatherProof_combine'
img_norm_cfg = dict(
    mean=[103.336, 104.443, 100.035], 
    std=[39.329, 38.147, 42.803], 
    to_rgb=True)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRealImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='My_Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='My_RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='My_RandomFlip', prob=0.5),
    dict(type='My_PhotoMetricDistortion'),
    dict(type='My_Normalize', **img_norm_cfg),
    dict(type='My_Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='My_DefaultFormatBundle'),
    dict(type='My_Collect', keys=['img', 'real_img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        real_img_dir='real_images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=test_pipeline))
