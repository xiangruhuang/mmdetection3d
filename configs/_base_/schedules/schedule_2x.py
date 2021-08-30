# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=1e-2)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
