# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='none',
    warmup_iters=None,
    warmup_ratio=None,
    step=1000,
    gamma=0.9)
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
