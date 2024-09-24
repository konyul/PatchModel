_base_ = [
    '../_base_/models/patchnet_resnet34.py',
    '../_base_/datasets/hyundae_w_aug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
dataset_type = 'HyundaeDataset'
data_root = 'data/hyundae/'
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

# load_from = './work_dirs/hmc_5000_pretrained_conv3x3_dilated_x2/iter_36000.pth'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=20),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='img_dir/demo_1_5', seg_map_path='ann_dir/demo_1_5'),
            img_path='img_dir/blur_sampled_train', seg_map_path='ann_dir/blur_sampled_train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='img_dir/demo_1_5', seg_map_path='ann_dir/demo_1_5'),
            img_path='img_dir/val_2nd/', seg_map_path='ann_dir/val_2nd/'),
            #img_path='img_dir/val_classified/non_WD', seg_map_path='ann_dir/val_classified/non_WD'),

        pipeline=test_pipeline))
test_dataloader = val_dataloader