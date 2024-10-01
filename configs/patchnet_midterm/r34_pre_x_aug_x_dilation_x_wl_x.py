_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py'
]
# dataset settings
dataset_type = 'HyundaeDataset'
data_root = 'data/hyundae/'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomRotate', prob=0.5, degree=20),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='img_dir/demo_1_5', seg_map_path='ann_dir/demo_1_5'),
            img_path='img_dir/train_2nd', seg_map_path='ann_dir/train_2nd'),
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

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

class_weight = [0.4, 30, 1.0]
# model settings
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='Patch_singlehead_EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        norm_cfg=norm_cfg,
        norm_eval=False,
        # dilations=(1, 3, 5, 7),
        style='pytorch',
        init_cfg=None
        ),
    decode_head=dict(
        type='PatchnetSingleHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        seg_head=True,
        channels=512,
        dropout_ratio=0.1,
        conv_next=True,
        num_classes=3,
        conv_kernel_size=7,
        conv_next_input_size=16,
        norm_cfg=norm_cfg,
        align_corners=False,
        attn=True,
        input_transform='multiple_select',
        init_cfg=None,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=(512, 512), stride=(768, 768)))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=0.),
            'backbone' : dict(lr_mult=0.1)
        }))
max_iters = 40000
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=max_iters,
        by_epoch=False,
    )
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

#load_from = '/mnt/4tb/hyundai/PatchModel/work_dirs/0723_woodscape_final/iter_20000.pth'
