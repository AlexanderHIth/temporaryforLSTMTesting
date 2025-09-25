#!/usr/bin/env python3

from pathlib import Path
import hvplot.pandas  # noqa: F401
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
# from PFCS.scripts.gt_plot import read_data
import imageio.v3 as iio
from PIL import Image
from moviepy import VideoFileClip
from rosbags.typesys import Stores, get_typestore
from rosbags.highlevel import AnyReader
import cv2
import warnings
import datetime as dt
import json
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# PRIMARY_COLOR = "#0072B5"
# SECONDARY_COLOR = "#B54300"
# CSV_FILE = (
#     "https://raw.githubusercontent.com/holoviz/panel/main/examples/assets/occupancy.csv"
# )
# hv.extension('bokeh')

DATA_PATH_ROOT = Path("/home/kir0ul/Projects/table-task-ur5e/")
BAG_FILE = DATA_PATH_ROOT / "rosbag2_2025-09-08_19-46-18_2025-09-08-19-46-19.bag"
GROUND_TRUTH_SEGM_FILE = DATA_PATH_ROOT / "table_task_UR5e_ground_truth.json"

pn.extension(design="material", sizing_mode="stretch_width")


def get_ground_truth_segmentation(ground_truth_segm_file, bagfile):
    if not ground_truth_segm_file.exists():
        print(
            "JSON ground truth segmentation file not found:\n"
            f"`{ground_truth_segm_file}`"
        )
        return
    with open(ground_truth_segm_file) as fid:
        json_str = fid.read()
    gt_segm_all = json.loads(json_str)
    gt_segm_dict = None
    for item in gt_segm_all:
        if item.get("filename") == bagfile.name:
            gt_segm_dict = item
            break
    if gt_segm_dict is None:
        print(f"Segmentation data not found in `{ground_truth_segm_file}`")
    return gt_segm_dict


def get_img_height_width(bagfile):
    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagfile], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == "/imu_raw/Imu"]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            # print(msg.header.frame_id
            if connection.msgtype == "sensor_msgs/msg/Image":
                # print(msg)
                break
    return msg.height, msg.width, msg.data


def extract_video_from_bag(bagfile, fps=20):
    print("Extracting video from Bag file...")

    extension = "mkv"
    video_path = bagfile.parent / (bagfile.stem + "." + extension)

    # Initialize video writer
    img_height, img_width, _ = get_img_height_width(bagfile=bagfile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for MP4 format
    video = cv2.VideoWriter(
        filename=video_path, fourcc=fourcc, fps=fps, frameSize=(img_width, img_height)
    )

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
            if connection.msgtype == "sensor_msgs/msg/Image":
                frame = msg.data.reshape((msg.height, msg.width, 3))

                current_ts = pd.to_datetime(timestamp, utc=True).tz_convert("EST")

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Add timestamp overlay on image
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt1 = current_ts.isoformat()
                txt2 = str(current_ts.timestamp())
                fontScale = 0.5
                white = (255, 255, 255)
                fontthickness = 1
                cv2.putText(
                    img=img,
                    text=txt1,
                    org=(10, 30),
                    fontFace=font,
                    fontScale=fontScale,
                    color=white,
                    thickness=fontthickness,
                )
                cv2.putText(
                    img=img,
                    text=txt2,
                    org=(10, 50),
                    fontFace=font,
                    fontScale=fontScale,
                    color=white,
                    thickness=fontthickness,
                )

                # Add images to the video
                video.write(img)

    # Release the video writer
    video.release()
    cv2.destroyAllWindows()
    print("Extracting video from Bag file: done ✓")
    print(f"Video path: {video_path}")
    return video_path


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


traj = extract_eef_data_from_rosbag(BAG_FILE)


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


gt_segm_dict = get_ground_truth_segmentation(
    ground_truth_segm_file=GROUND_TRUTH_SEGM_FILE, bagfile=BAG_FILE
)
gt_file_keys = set(gt_segm_dict.keys())
gt_file_keys.remove("filename")
gt_file_keys.add("")
skill_choice_widget = pn.widgets.Select(
    name="Skill", value="", options=list(gt_file_keys)
)
video_path = extract_video_from_bag(bagfile=BAG_FILE, fps=20)
clip = VideoFileClip(video_path)
frames_count = clip.reader.n_frames - 1
epoch_ini = int(traj.timestamp.iloc[0].timestamp()) + 1
epoch_end = int(traj.timestamp.iloc[-1].timestamp())
slider_widget = pn.widgets.IntSlider(
    name="Epoch",
    value=int((epoch_end - epoch_ini) / 2 + epoch_ini),
    start=epoch_ini,
    end=epoch_end - 1,
)


def get_frame_plot(epoch_req, epoch_ini):
    idx = epoch_req - epoch_ini
    img = get_video_frame(
        index=idx,
        video_path=video_path,
    )
    frame_plot = pn.pane.Image(Image.fromarray(img), width=480, align="center")
    return frame_plot


line_plt = pn.bind(
    get_line_plot,
    traj=traj,
    epoch_req=slider_widget,
    skill_choice=skill_choice_widget,
    gt_segm_dict=gt_segm_dict,
)
img_plt = pn.bind(
    get_frame_plot,
    epoch_req=slider_widget,
    epoch_ini=epoch_ini,
)

centered_img = pn.Row(pn.layout.HSpacer(), img_plt, pn.layout.HSpacer())


pn.template.MaterialTemplate(
    site="Segmentation",
    title="Video vs. end effector position",
    sidebar=[skill_choice_widget],
    main=[centered_img, slider_widget, line_plt],
).servable()  # The ; is needed in the notebook to not display the template. Its not needed in a script
