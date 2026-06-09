# SPDX-License-Identifier: Apache-2.0
from torchvision import transforms
from torchvision.transforms import Lambda

from fastvideo.dataset.parquet_dataset_map_style import (
    build_parquet_map_style_dataloader, )
from fastvideo.dataset.preprocessing_datasets import (
    PreprocessBatch,
    TextDataset,
    VideoCaptionMergedDataset,
)
from fastvideo.dataset.transform import (
    CenterCropResizeVideo,
    Normalize255,
    TemporalRandomCrop,
)


def getdataset(args) -> VideoCaptionMergedDataset:
    temporal_sample = TemporalRandomCrop(args.num_frames) if args.do_temporal_sample else None
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    resize_topcrop = [
        CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True),
    ]
    resize = [
        CenterCropResizeVideo((args.max_height, args.max_width)),
    ]
    transform = transforms.Compose([
        *resize,
    ])
    transform_topcrop = transforms.Compose([
        Normalize255(),
        *resize_topcrop,
        norm_fun,
    ])
    return VideoCaptionMergedDataset(
        data_merge_path=args.data_merge_path,
        args=args,
        transform=transform,
        temporal_sample=temporal_sample,
        transform_topcrop=transform_topcrop,
        seed=args.seed,
    )


def gettextdataset(args) -> TextDataset:
    return TextDataset(
        data_merge_path=args.data_merge_path,
        args=args,
        seed=args.seed,
    )


__all__ = [
    "PreprocessBatch",
    "TextDataset",
    "VideoCaptionMergedDataset",
    "build_parquet_map_style_dataloader",
    "getdataset",
    "gettextdataset",
]
