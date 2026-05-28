auto_scale_lr = dict(base_batch_size=80)
backend_args = dict(backend='local')
codec = dict(
    decode_max_instances=30,
    generate_keypoint_heatmaps=True,
    heatmap_size=(
        128,
        128,
    ),
    input_size=(
        512,
        512,
    ),
    minimal_diagonal_length=5.656854249492381,
    sigma=(
        4,
        2,
    ),
    type='SPR')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'bottomup'
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    32,
                    64,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    32,
                    64,
                    128,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                multiscale_output=True,
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    32,
                    64,
                    128,
                    256,
                ),
                num_modules=3)),
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth',
            type='Pretrained'),
        type='HRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            decode_max_instances=30,
            generate_keypoint_heatmaps=True,
            heatmap_size=(
                128,
                128,
            ),
            input_size=(
                512,
                512,
            ),
            minimal_diagonal_length=5.656854249492381,
            sigma=(
                4,
                2,
            ),
            type='SPR'),
        displacement_loss=dict(
            beta=0.1111111111111111,
            loss_weight=0.002,
            supervise_empty=False,
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True),
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        in_channels=480,
        num_keypoints=17,
        rescore_cfg=dict(
            in_channels=74,
            init_cfg=dict(
                checkpoint=
                'https://download.openmmlab.com/mmpose/pretrain_models/kpt_rescore_coco-33d58c5c.pth',
                type='Pretrained'),
            norm_indexes=(
                5,
                6,
            )),
        type='DEKRHead'),
    neck=dict(concat=True, type='FeatureMapProcessor'),
    test_cfg=dict(
        align_corners=False,
        flip_test=True,
        multiscale_test=False,
        nms_dist_thr=0.05,
        shift_heatmap=True),
    type='BottomupPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.001, type='Adam'))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=140,
        gamma=0.1,
        milestones=[
            90,
            120,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImage'),
            dict(
                input_size=(
                    512,
                    512,
                ),
                resize_mode='expand',
                size_factor=32,
                type='BottomupResize'),
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'skeleton_links',
                ),
                type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='keypoint',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=140, val_interval=10)
train_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file='annotations/person_keypoints_train2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImage'),
            dict(input_size=(
                512,
                512,
            ), type='BottomupRandomAffine'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                encoder=dict(
                    decode_max_instances=30,
                    generate_keypoint_heatmaps=True,
                    heatmap_size=(
                        128,
                        128,
                    ),
                    input_size=(
                        512,
                        512,
                    ),
                    minimal_diagonal_length=5.656854249492381,
                    sigma=(
                        4,
                        2,
                    ),
                    type='SPR'),
                type='GenerateTarget'),
            dict(type='BottomupGetHeatmapMask'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(input_size=(
        512,
        512,
    ), type='BottomupRandomAffine'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        encoder=dict(
            decode_max_instances=30,
            generate_keypoint_heatmaps=True,
            heatmap_size=(
                128,
                128,
            ),
            input_size=(
                512,
                512,
            ),
            minimal_diagonal_length=5.656854249492381,
            sigma=(
                4,
                2,
            ),
            type='SPR'),
        type='GenerateTarget'),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImage'),
            dict(
                input_size=(
                    512,
                    512,
                ),
                resize_mode='expand',
                size_factor=32,
                type='BottomupResize'),
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'crowd_index',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                    'flip',
                    'flip_direction',
                    'flip_indices',
                    'raw_ann_info',
                    'skeleton_links',
                ),
                type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='keypoint',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        input_size=(
            512,
            512,
        ),
        resize_mode='expand',
        size_factor=32,
        type='BottomupResize'),
    dict(
        meta_keys=(
            'id',
            'img_id',
            'img_path',
            'crowd_index',
            'ori_shape',
            'img_shape',
            'input_size',
            'input_center',
            'input_scale',
            'flip',
            'flip_direction',
            'flip_indices',
            'raw_ann_info',
            'skeleton_links',
        ),
        type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
