#!/usr/bin/env python3

from pathlib import Path
import hvplot.pandas  # noqa: F401
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from PFCS.scripts.gt_plot import read_data
import imageio.v3 as iio
from PIL import Image
from moviepy import VideoFileClip
from rosbags.typesys import Stores, get_typestore
from rosbags.highlevel import AnyReader
import cv2

# PRIMARY_COLOR = "#0072B5"
# SECONDARY_COLOR = "#B54300"
# CSV_FILE = (
#     "https://raw.githubusercontent.com/holoviz/panel/main/examples/assets/occupancy.csv"
# )
# hv.extension('bokeh')

# VIDEO_PATH = "/home/kir0ul/Projects/TableTaskVideos/2.webm"
DATA_PATH_ROOT = Path("/home/kir0ul/Projects/table-task-ur5e/")
VIDEO_PATH = DATA_PATH_ROOT / "rosbag2_2025-09-08_19-46-18_2025-09-08-19-46-19.mkv"
BAG_FILE = DATA_PATH_ROOT / "rosbag2_2025-09-08_19-46-18_2025-09-08-19-46-19.bag"
SKILL_CHOICE = ["", "Reaching", "Placing"]

pn.extension(design="material", sizing_mode="stretch_width")


@pn.cache
def get_skill_data(filenum, skill):
    skill_data = [
        {
            SKILL_CHOICE[1]: [
                {"ini": 133, "end": 1242},
            ],
            SKILL_CHOICE[2]: [
                {"ini": 1403, "end": 1602},
            ],
        },
        {
            SKILL_CHOICE[1]: [
                {"ini": 133, "end": 1242},
                {"ini": 1694, "end": 3239},
                {"ini": 3802, "end": 5138},
                {"ini": 5737, "end": 6158},
                {"ini": 7067, "end": 7478},
            ],
            SKILL_CHOICE[2]: [
                {"ini": 1403, "end": 1602},
                {"ini": 3383, "end": 3680},
                {"ini": 5246, "end": 5595},
                {"ini": 6517, "end": 6899},
                {"ini": 7573, "end": 7850},
            ],
        },
    ]
    return skill_data[filenum][skill]


# video = pn.pane.Video(
#     "/home/kir0ul/Projects/TableTaskVideos/2.webm", width=720, loop=False
# )
def get_video_frame(index, video_path):
    # read a single frame
    try:
        frame = iio.imread(
            video_path,
            index=index,
            plugin="pyav",
        )
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return img
    except StopIteration:
        print("Reached the end of the video file")
        return np.asarray(Image.new("RGB", (3840, 2160), (0, 0, 0)))
        # return np.asarray(Image.new("RGB", (720, 405), (0, 0, 0)))


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
        for connection, timestamp, rawdata in reader.messages(connections=connections):
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
    print("Extracting TF & gripper data from Bag file: done ✓")
    return tf_df, gripper_df


tf_df, gripper_df = extract_eef_data_from_rosbag(BAG_FILE)


def get_line_plot(tf_df, gripper_df, frame_idx, skill_choice=None):
    # vline = hv.VLine(df.timestamps[frame_idx]).opts(
    #     color="black", line_dash="dashed", line_width=6
    # )
    vline = hv.VLine(frame_idx).opts(color="black", line_dash="dashed", line_width=3)
    # print(f"\nTimestamp slider: {df.timestamps[frame_idx]}\n")
    # lineplot_tf = df.hvplot(x="timestamps", y=["x", "y", "z"], height=400)
    lineplot_tf = tf_df.hvplot(x="timestamp", y=["x", "y", "z"], height=400).opts(
        xlabel="Time", ylabel="Position"
    )
    lineplot_grip = gripper_df.hvplot(x="timestamp", y=["val"], label="gripper")
    # overlay.opts(opts.VLine(color="red", line_dash='dashed', line_width=6))
    overlay = lineplot_tf * lineplot_grip * vline
    fill_min = np.min([tf_df.x.min(), tf_df.y.min(), tf_df.z.min()])
    fill_max = np.max([tf_df.x.max(), tf_df.y.max(), tf_df.z.max()])

    if skill_choice != "":
        skill_data = get_skill_data(filenum=FILENUM, skill=skill_choice)
        for sect_i, sect_val in enumerate(skill_data):
            xs = tf_df.index[sect_val["ini"] : sect_val["end"]]
            spread = hv.Spread(
                (
                    xs,
                    fill_max - fill_min,
                    fill_min - 2,
                    fill_max + 2,
                ),
                # label=sect_key,
            ).opts(fill_alpha=0.15, color="gray")
            overlay = overlay * spread

    # else:
    #     for sect_i, sect_key in enumerate(file_ground_truth["idx"].keys()):
    #         sect_dict_current = file_ground_truth["idx"][sect_key]
    #         xs = tf_df.index[sect_dict_current["ini"] : sect_dict_current["end"]]
    #         spread = hv.Spread(
    #             (
    #                 xs,
    #                 fill_max - fill_min,
    #                 fill_min - 2,
    #                 fill_max + 2,
    #             ),
    #             label=sect_key,
    #             # vdims=["y", "yerrneg", "yerrpos"],
    #         ).opts(fill_alpha=0.15)
    #         overlay = overlay * spread
    return overlay.opts(ylim=(fill_min - 0.1, fill_max + 0.1))


skill_choice_widget = pn.widgets.Select(name="Skill", value="", options=SKILL_CHOICE)
clip = VideoFileClip(VIDEO_PATH)
frame_count = clip.reader.n_frames - 1
slider_widget = pn.widgets.IntSlider(
    name="Index", value=int(len(tf_df) / 2), start=0, end=len(tf_df)
)


def get_frame_plot(frame_idx, frame_count, plot_pts_num):
    idx = int(frame_count * frame_idx / plot_pts_num)
    img = get_video_frame(
        index=idx,
        video_path=VIDEO_PATH,
    )
    frame_plot = pn.pane.Image(Image.fromarray(img), width=480, align="center")
    return frame_plot


line_plt = pn.bind(
    get_line_plot,
    tf_df=tf_df,
    gripper_df=gripper_df,
    frame_idx=slider_widget,
    skill_choice=skill_choice_widget,
)
img_plt = pn.bind(
    get_frame_plot,
    frame_idx=slider_widget,
    frame_count=frame_count,
    plot_pts_num=len(tf_df),
)

centered_img = pn.Row(pn.layout.HSpacer(), img_plt, pn.layout.HSpacer())


pn.template.MaterialTemplate(
    site="Segmentation",
    title="Video vs. end effector position",
    sidebar=[skill_choice_widget],
    main=[centered_img, slider_widget, line_plt],
).servable()  # The ; is needed in the notebook to not display the template. Its not needed in a script
