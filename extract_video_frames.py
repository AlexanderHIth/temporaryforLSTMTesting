#!/usr/bin/env python3

from tqdm.auto import tqdm
import imageio.v3 as iio
from pathlib import Path
from moviepy import VideoFileClip
import numpy as np
from segmentation_utils import extract_video_from_bag, get_video_frame

FRAMES_NB = 20
DATA_PATH_ROOT = Path("/home/kir0ul/Projects/table-task-ur5e/")
BAG_FILE = DATA_PATH_ROOT / "rosbag2_2025-09-08_19-46-18_2025-09-08-19-46-19.bag"
IMG_ROOT = DATA_PATH_ROOT / BAG_FILE.stem
IMG_ROOT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    video_path = extract_video_from_bag(bagfile=BAG_FILE, fps=20)
    clip = VideoFileClip(video_path)
    frames_count = clip.reader.n_frames - 1
    frames_idx2extract = np.linspace(
        start=0, stop=frames_count, num=FRAMES_NB, dtype=np.int_
    )

    for idx in tqdm(frames_idx2extract):
        img = get_video_frame(
            index=idx,
            video_path=video_path,
        )
        img_path = IMG_ROOT / f"{idx}.png"
        iio.imwrite(uri=img_path, image=img)
