#!/usr/bin/env python3

import pandas as pd
from tqdm.auto import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import json


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
