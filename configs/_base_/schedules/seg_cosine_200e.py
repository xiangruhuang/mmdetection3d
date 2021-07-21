# optimizer
# This schedule is mainly used on ScanNet dataset in segmentation task
optimizer = dict(type='Adam', lr=1e-4, weight_decay=1e-3)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-6)
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=250)
