_base_ = [
    '../_base_/models/patchnet_resnet34.py',
    '../_base_/datasets/hyundae_wo_aug.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')),
    test_cfg=dict(mode='whole', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=0.),
            'backbone' : dict(lr_mult=0.1)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=40000,
        by_epoch=False,
    )
]

#load_from = '/mnt/4tb/hyundai/PatchModel/work_dirs/0723_woodscape_final/iter_20000.pth'

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
