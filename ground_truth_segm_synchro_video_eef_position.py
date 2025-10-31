#!/usr/bin/env python3

from pathlib import Path
import hvplot.pandas  # noqa: F401
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

# from PFCS.scripts.gt_plot import read_data
from PIL import Image

# from moviepy import VideoFileClip
from rosbags.typesys import Stores, get_typestore
from rosbags.highlevel import AnyReader
import warnings
import datetime as dt
import json
from tqdm.auto import tqdm
from segmentation_utils import (
    get_ground_truth_segmentation,
    get_bagfiles_from_json,
    extract_video_from_bag,
    get_video_frame,
)

pn.extension()
warnings.filterwarnings("ignore", category=UserWarning)

# PRIMARY_COLOR = "#0072B5"
# SECONDARY_COLOR = "#B54300"
# CSV_FILE = (
#     "https://raw.githubusercontent.com/holoviz/panel/main/examples/assets/occupancy.csv"
# )
# hv.extension('bokeh')

# DATA_PATH_ROOT = Path("/home/kir0ul/Projects/table-task-ur5e/")
# GROUND_TRUTH_SEGM_FILE = DATA_PATH_ROOT / "table_task_UR5e_ground_truth.json"
GROUND_TRUTH_SEGM_FILE = Path(".") / "table_task_UR5e_ground_truth.json"
# BAG_FILE = DATA_PATH_ROOT / "rosbag2_2025-09-08_19-46-18_2025-09-08-19-46-19.bag"
BAGFILE_NUM = 0


pn.extension(design="material", sizing_mode="stretch_width")


def get_bagfiles_list(ground_truth_segm_file):
    if not ground_truth_segm_file.exists():
        print(
            "JSON ground truth segmentation file not found:\n"
            f"`{ground_truth_segm_file}`"
        )
        return

    # Load JSON as dict
    with open(ground_truth_segm_file) as fid:
        json_str = fid.read()
    json_str = json.loads(json_str)
    data_path_root = json_str.get("root_path")
    gt_array = json_str.get("groundtruth")
    bagfiles = [item.get("filename") for item in gt_array]
    return data_path_root, bagfiles


@pn.cache
def extract_eef_data_from_rosbag(bagfile):
    print("Extracting TF & gripper data from Bag file...")
    tf = {"x": [], "y": [], "z": [], "timestamp": []}
    gripper = {"val": [], "timestamp": []}

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagfile], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == "/imu_raw/Imu"]
        msg_nb_total = len(list(reader.messages(connections=connections)))
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=connections), total=msg_nb_total
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)
            # print(msg.header.frame_id
            if connection.msgtype == "tf2_msgs/msg/TFMessage":
                if (
                    msg.transforms[0].child_frame_id == "tool0_controller"
                    and msg.transforms[0].header.frame_id == "base"
                ):
                    tf["x"].append(msg.transforms[0].transform.translation.x)
                    tf["y"].append(msg.transforms[0].transform.translation.y)
                    tf["z"].append(msg.transforms[0].transform.translation.z)
                    # tf["timestamp"].append(pd.to_datetime(timestamp, utc=True).tz_localize("UTC").tz_convert("EST"))
                    tf["timestamp"].append(
                        pd.to_datetime(timestamp, utc=True).tz_convert("EST")
                    )

            if connection.msgtype == "ur5e_move/msg/gripper_pos":
                gripper["val"].append(msg.gripper_pos)
                gripper["timestamp"].append(
                    pd.to_datetime(timestamp, utc=True).tz_convert("EST")
                )

    tf_df = pd.DataFrame(tf)
    gripper_df = pd.DataFrame(gripper)
    gripper_df["val"] = gripper_df["val"].apply(lambda elem: elem / 100)
    #
    # Merge both DataFrames into one
    traj = pd.merge_asof(tf_df, gripper_df, on="timestamp")
    traj.dropna(inplace=True, ignore_index=True)
    traj.rename(columns={"val": "gripper"}, inplace=True)
    print("Extracting TF & gripper data from Bag file: done ✓")
    return traj


def get_line_plot(traj, epoch_req, gt_segm_dict=None, skill_choice=None):
    slider_ts = dt.datetime.fromtimestamp(epoch_req) - dt.timedelta(hours=1)
    vline = hv.VLine(slider_ts).opts(color="black", line_dash="dashed", line_width=3)
    lineplot_tf = traj.hvplot(x="timestamp", y=["x", "y", "z"], height=400).opts(
        xlabel="Time", ylabel="Position"
    )
    lineplot_grip = traj.hvplot(x="timestamp", y=["gripper"], label="gripper")
    # overlay.opts(opts.VLine(color="red", line_dash='dashed', line_width=6))
    overlay = lineplot_tf * lineplot_grip * vline
    fill_min = np.min([traj.x.min(), traj.y.min(), traj.z.min()])
    fill_max = np.max([traj.x.max(), traj.y.max(), traj.z.max()])

    if skill_choice == "HigherLevel":
        for sect_key in gt_segm_dict[skill_choice]:
            sect_val = gt_segm_dict[skill_choice][sect_key]
            xs = traj.timestamp[
                (
                    traj.timestamp
                    > pd.Timestamp(
                        dt.datetime.fromtimestamp(sect_val["ini"])
                        - dt.timedelta(hours=1),
                        tz="EST",
                    )
                )
                & (
                    traj.timestamp
                    < pd.Timestamp(
                        dt.datetime.fromtimestamp(sect_val["end"])
                        - dt.timedelta(hours=1),
                        tz="EST",
                    )
                )
            ] - dt.timedelta(hours=5)
            spread = hv.Spread(
                (
                    xs,
                    fill_max - fill_min,
                    fill_min - 2,
                    fill_max + 2,
                ),
                label=sect_key,
            ).opts(fill_alpha=0.15)
            overlay = overlay * spread

    elif skill_choice == "LowerLevel":
        palette = hv.Palette.default_cycles["Set1"]
        for idx, sect_key in enumerate(gt_segm_dict[skill_choice]):
            sect_val = gt_segm_dict[skill_choice][sect_key]
            for sect_cur in sect_val:
                xs = traj.timestamp[
                    (
                        traj.timestamp
                        > pd.Timestamp(
                            dt.datetime.fromtimestamp(sect_cur["ini"])
                            - dt.timedelta(hours=1),
                            tz="EST",
                        )
                    )
                    & (
                        traj.timestamp
                        < pd.Timestamp(
                            dt.datetime.fromtimestamp(sect_cur["end"])
                            - dt.timedelta(hours=1),
                            tz="EST",
                        )
                    )
                ] - dt.timedelta(hours=5)
                spread = hv.Spread(
                    (
                        xs,
                        fill_max - fill_min,
                        fill_min - 2,
                        fill_max + 2,
                    ),
                    label=sect_key,
                ).opts(fill_alpha=0.15, color=palette[idx])
                overlay = overlay * spread

    return overlay.opts(ylim=(fill_min - 0.1, fill_max + 0.1))


@pn.cache
def get_frame_plot(epoch_req, epoch_ini):
    idx = epoch_req - epoch_ini
    img = get_video_frame(
        index=idx,
        video_path=video_path,
    )
    frame_plot = pn.pane.Image(Image.fromarray(img), width=480, align="center")
    return frame_plot


data_path_root, bagfiles = get_bagfiles_list(
    ground_truth_segm_file=GROUND_TRUTH_SEGM_FILE
)


bagfiles = get_bagfiles_from_json(ground_truth_segm_file=GROUND_TRUTH_SEGM_FILE)
bagfile = bagfiles[BAGFILE_NUM]

gt_segm_dict = get_ground_truth_segmentation(
    ground_truth_segm_file=GROUND_TRUTH_SEGM_FILE, bagfile=bagfile
)
gt_file_keys = set(gt_segm_dict.keys())
gt_file_keys.remove("filename")
gt_file_keys.add("")
skill_choice_dropdown = pn.widgets.Select(
    name="Skill", value="", options=list(gt_file_keys)
)
video_path = extract_video_from_bag(bagfile=bagfile, fps=20)
traj = extract_eef_data_from_rosbag(bagfile=bagfile)
epoch_ini = int(traj.timestamp.iloc[0].timestamp()) + 1
epoch_end = int(traj.timestamp.iloc[-1].timestamp())
slider_widget = pn.widgets.IntSlider(
    name="Epoch",
    value=int((epoch_end - epoch_ini) / 2 + epoch_ini),
    start=epoch_ini,
    end=epoch_end - 1,
)


# Widgets & web app logic
line_plt = pn.bind(
    get_line_plot,
    traj=traj,
    epoch_req=slider_widget,
    skill_choice=skill_choice_dropdown,
    gt_segm_dict=gt_segm_dict,
)
img_plt = pn.bind(
    get_frame_plot,
    epoch_req=slider_widget,
    epoch_ini=epoch_ini,
)
# json_input = pn.widgets.FileInput()
# bagfiles_choice_dropdown = pn.widgets.Select(
#     name="Bag file", value="", options=bagfiles
# )
centered_img = pn.Row(pn.layout.HSpacer(), img_plt, pn.layout.HSpacer())
pn.template.MaterialTemplate(
    site="Segmentation",
    title="Video vs. end effector position",
    # sidebar=[json_input, bagfiles_choice_dropdown, skill_choice_dropdown],
    sidebar=[skill_choice_dropdown],
    main=[centered_img, slider_widget, line_plt],
).servable()  # The ; is needed in the notebook to not display the template. Its not needed in a script
