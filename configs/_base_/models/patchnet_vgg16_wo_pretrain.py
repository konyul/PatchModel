# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='Patch_singlehead_EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='VGG',
        depth=16,
        with_bn=False,
        num_stages=5,
        dilations=(1, 1, 1, 1, 1),
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        bn_eval=False,
        bn_frozen=False,
        ceil_mode=False,
        #DY with_last_pool False -> True
        with_last_pool=True,
    ),
    decode_head=dict(
        type='PatchnetSingleHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        seg_head=True,
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        conv_kernel_size='multi',
        norm_cfg=norm_cfg,
        align_corners=False,
        input_transform='multiple_select',
        init_cfg=None,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))