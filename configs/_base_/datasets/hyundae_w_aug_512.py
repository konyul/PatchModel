# dataset settings
dataset_type = 'HyundaeDataset'
data_root = 'data/hyundae/'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=20),
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
    batch_size=2,
    num_workers=2,
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
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='img_dir/demo_1_5', seg_map_path='ann_dir/demo_1_5'),
            img_path='img_dir/val_2nd', seg_map_path='ann_dir/val_2nd'),
            #img_path='img_dir/val_classified/non_WD', seg_map_path='ann_dir/val_classified/non_WD'),

        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
