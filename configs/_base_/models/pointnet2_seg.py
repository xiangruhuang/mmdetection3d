model=dict(
    type='PointNet2Seg',
    backbone=dict(
        type='PointNet2SASSG',
        num_points=(32, 16, 8, 4),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256, 256), (256, 256, 256),
                     (256, 256, 128), (128, 128, 128, 128)),
        in_channels=3,
        ),
    decode_head=dict(
        type='PointNet2Head',
        channels=128,
        num_classes=2,
        dropout_ratio=0.0,
        ),
)
