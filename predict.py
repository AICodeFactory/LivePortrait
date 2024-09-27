# coding: utf-8
"""
for human prediction
"""

import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from cog import BasePredictor, Input, Path

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"FFmpeg check failed: {e}")
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"Source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"Driving info not found: {args.driving}")

class Predictor(BasePredictor):
    def setup(self):
        # Set up the environment for FFmpeg
        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

        if not fast_check_ffmpeg():
            raise ImportError(
                "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script."
            )

    def predict(
        self,
        source: Path = Input(
            description="Source image or video file"
        ),
        driving: Path = Input(
            description="Driving image or video file"
        )
    ) -> list[Path]:

        # 组装ArgumentConfig
        args = ArgumentConfig(source=str(source), driving=str(driving))

        # Specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )

        # Run the prediction
        wfp, wfp_concat = live_portrait_pipeline.execute(args)

        return [Path(wfp), Path(wfp_concat)]
