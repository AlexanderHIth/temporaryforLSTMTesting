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

# PRIMARY_COLOR = "#0072B5"
# SECONDARY_COLOR = "#B54300"
# CSV_FILE = (
#     "https://raw.githubusercontent.com/holoviz/panel/main/examples/assets/occupancy.csv"
# )
# hv.extension('bokeh')

# DATA_PATH = "/home/kir0ul/Projects/table-task-ur5e/rosbag2_2025-09-08_16-33-22_2025-09-08-16-33-23.bag"

VIDEO_PATH = "/home/kir0ul/Projects/TableTaskVideos/2.webm"
FILENUM = 1
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
        return frame
    except StopIteration:
        print("Reached the end of the video file")
        return np.asarray(Image.new("RGB", (3840, 2160), (0, 0, 0)))
        # return np.asarray(Image.new("RGB", (720, 405), (0, 0, 0)))


@pn.cache
def get_table_task_data(filenum, sect_key="fork"):
    task_ground_truth = [
        {
            "filename": "fetch_recorded_demo_1730997119",
            "idx": {
                "plate": {"ini": 0, "end": 1125},
                "napkin": {"ini": 1125, "end": 2591},
                "cup": {"ini": 2591, "end": 3986},
                "fork": {"ini": 3986, "end": 5666},
                "spoon": {"ini": 5666, "end": 7338},
            },
        },
        {
            "filename": "fetch_recorded_demo_1730997530",
            "idx": {
                "plate": {"ini": 0, "end": 1812},
                "napkin": {"ini": 1812, "end": 3844},
                "cup": {"ini": 3844, "end": 5732},
                "fork": {"ini": 5732, "end": 7090},
                "spoon": {"ini": 7090, "end": 7955},
            },
        },
        {
            "filename": "fetch_recorded_demo_1730997735",
            "idx": {
                "plate": {"ini": 0, "end": 1965},
                "napkin": {"ini": 1965, "end": 4178},
                "cup": {"ini": 4178, "end": 6427},
                "spoon": {"ini": 6427, "end": 7904},
                "fork": {"ini": 7904, "end": 9123},
            },
        },
        {
            "filename": "fetch_recorded_demo_1730997956",
            "idx": {
                "plate": {"ini": 0, "end": 1898},
                "napkin": {"ini": 1898, "end": 4081},
                "cup": {"ini": 4081, "end": 5442},
                "spoon": {"ini": 5442, "end": 6829},
                "fork": {"ini": 6829, "end": 9177},
            },
        },
    ]

    datapath_root = Path("./PFCS/table task")
    xyz_path = (
        datapath_root
        / "xyz data"
        / "full_tasks"
        / (task_ground_truth[filenum]["filename"] + ".txt")
    )
    h5_path = (
        datapath_root / "h5 files" / (task_ground_truth[filenum]["filename"] + ".h5")
    )

    data = np.loadtxt(xyz_path)  # load the file into an array
    joint_data, tf_data, gripper_data = read_data(h5_path)

    time_sec = tf_data[0][:, 0]
    time_nanosec = tf_data[0][:, 1]

    timestamps = []
    for t_idx, t_val in enumerate(time_sec):
        timestamp = pd.Timestamp(time_sec[t_idx], unit="s", tz="EST") + pd.to_timedelta(
            time_nanosec[t_idx], unit="ns"
        )
        timestamps.append(timestamp)
    timestamps = pd.Series(timestamps)

    traj_df = pd.DataFrame(
        {"x": data[:, 0], "y": data[:, 1], "z": data[:, 2], "timestamps": timestamps}
    )

    # sect_dict_current = task_ground_truth[filenum]["idx"][sect_key]
    # return traj_df[sect_dict_current["ini"] : sect_dict_current["end"]]
    return traj_df, task_ground_truth[filenum]


data_df, file_ground_truth = get_table_task_data(filenum=FILENUM)


def get_line_plot(df, frame_idx, skill_choice=None):
    # vline = hv.VLine(df.timestamps[frame_idx]).opts(
    #     color="black", line_dash="dashed", line_width=6
    # )
    vline = hv.VLine(frame_idx).opts(color="black", line_dash="dashed", line_width=3)
    # print(f"\nTimestamp slider: {df.timestamps[frame_idx]}\n")
    # lineplot = df.hvplot(x="timestamps", y=["x", "y", "z"], height=400)
    lineplot = df.hvplot(x="index", y=["x", "y", "z"], height=400).opts(
        xlabel="Index", ylabel="Position"
    )
    # overlay.opts(opts.VLine(color="red", line_dash='dashed', line_width=6))
    overlay = lineplot * vline
    fill_min = np.min([df.x.min(), df.y.min(), df.z.min()])
    fill_max = np.max([df.x.max(), df.y.max(), df.z.max()])

    if skill_choice != "":
        skill_data = get_skill_data(filenum=FILENUM, skill=skill_choice)
        for sect_i, sect_val in enumerate(skill_data):
            xs = df.index[sect_val["ini"] : sect_val["end"]]
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

    else:
        for sect_i, sect_key in enumerate(file_ground_truth["idx"].keys()):
            sect_dict_current = file_ground_truth["idx"][sect_key]
            xs = df.index[sect_dict_current["ini"] : sect_dict_current["end"]]
            spread = hv.Spread(
                (
                    xs,
                    fill_max - fill_min,
                    fill_min - 2,
                    fill_max + 2,
                ),
                label=sect_key,
                # vdims=["y", "yerrneg", "yerrpos"],
            ).opts(fill_alpha=0.15)
            overlay = overlay * spread
    return overlay.opts(ylim=(fill_min - 0.1, fill_max + 0.1))


skill_choice_widget = pn.widgets.Select(name="Skill", value="", options=SKILL_CHOICE)
clip = VideoFileClip(VIDEO_PATH)
frame_count = clip.reader.n_frames - 1
slider_widget = pn.widgets.IntSlider(
    name="Index", value=int(len(data_df) / 2), start=0, end=len(data_df)
)


def get_frame_plot(frame_idx, frame_count, plot_pts_num):
    idx = int(frame_count * frame_idx / plot_pts_num)
    img = get_video_frame(
        index=idx,
        video_path=VIDEO_PATH,
    )
    frame_plot = pn.pane.Image(Image.fromarray(img), width=720, align="center")
    return frame_plot


line_plt = pn.bind(
    get_line_plot, df=data_df, frame_idx=slider_widget, skill_choice=skill_choice_widget
)
img_plt = pn.bind(
    get_frame_plot,
    frame_idx=slider_widget,
    frame_count=frame_count,
    plot_pts_num=len(data_df),
)

centered_img = pn.Row(pn.layout.HSpacer(), img_plt, pn.layout.HSpacer())


pn.template.MaterialTemplate(
    site="Segmentation",
    title="Video vs. end effector position",
    sidebar=[skill_choice_widget],
    main=[centered_img, slider_widget, line_plt],
).servable()  # The ; is needed in the notebook to not display the template. Its not needed in a script
