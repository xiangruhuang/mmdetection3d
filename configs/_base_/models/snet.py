# model settings
model = dict(
    type='SNet',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=5,
        num_points=(4096, 512, 512, 512),
        radius=(0.5, 1.0, 2.0, 4.0),
        num_samples=(128, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256, 256), (256, 256, 256), (256, 256, 256), (128, 128, 128)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    seg_head=dict(
        type='PointNet2Head',
        channels=128,
        num_classes=4,
        dropout_ratio=0.0,
        ),
    
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote'),
    test_cfg=dict(
        sample_mod='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True)
)
