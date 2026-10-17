"""
uv run ./launch/data_make_action_transaction.py \
    --imitation_data_save_path ./datas/reach_lift_imitation_action_chunk \
    --data_load_path ./datas/reach_lift_data1 \
    --action_chunk_num 16 \
    --detail_str act_ep_* \
    --max_episode 150000
"""

import os
import argparse
import numpy as np
# e+1 이나 e-1 이렇게 표기를 없애줌
np.set_printoptions(suppress=True)
import glob
import shutil

# --------- args ---------
parser = argparse.ArgumentParser(description="imitation action chunck")
parser.add_argument(
    "--imitation_data_save_path",
    type=str,
    default="./datas",
)
parser.add_argument(
    "--data_load_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--action_chunk_num",
    type=int,
    default=8,
)
parser.add_argument(
    "--detail_str",
    type=str,
    default=None,
)
parser.add_argument(
    "--max_episode",
    type=int,
    default=None,
)

def make_action_chunks(data, chunk_size=16):
    T, D = data.shape
    chunks = np.empty((T, chunk_size, D), dtype=data.dtype)

    for t in range(T):
        end = min(t + chunk_size, T)
        valid = data[t:end]
        chunks[t, :len(valid)] = valid

        if len(valid) < chunk_size:
            chunks[t, len(valid):] = valid[-1]

    return chunks


def organize_data_files(data_load_path):
    actions_dir = os.path.join(data_load_path, "actions")
    obs_dir = os.path.join(data_load_path, "obs")
    os.makedirs(actions_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)

    for path in glob.glob(os.path.join(data_load_path, "*act_*")):
        if os.path.isfile(path):
            shutil.move(path, os.path.join(actions_dir, os.path.basename(path)))

    for path in glob.glob(os.path.join(data_load_path, "*obs_*")):
        if os.path.isfile(path):
            shutil.move(path, os.path.join(obs_dir, os.path.basename(path)))


def get_load_paths(data_load_path, detail_str=None):
    if detail_str:
        if "act" in detail_str:
            target_dir = os.path.join(data_load_path, "actions")
            pattern = os.path.join(target_dir, detail_str)
        elif "obs" in detail_str:
            target_dir = os.path.join(data_load_path, "obs")
            pattern = os.path.join(target_dir, detail_str)
        else:
            pattern = os.path.join(data_load_path, "**", detail_str)

        return sorted(path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path))

    return sorted(
        path
        for path in glob.glob(os.path.join(data_load_path, "**", "*"), recursive=True)
        if os.path.isfile(path)
    )


if __name__ == "__main__":
    args = parser.parse_args()
    
    if not os.path.exists(args.data_load_path):
        print("로드할 파일이 존재하지 않음")
        
    else:
        organize_data_files(args.data_load_path)
        lists = get_load_paths(args.data_load_path, args.detail_str)

        os.makedirs(
            args.imitation_data_save_path,
            exist_ok=True
        )

        if args.max_episode:
            load_paths = lists[:args.max_episode]
        else:
            load_paths = lists

        for i in load_paths:
            save_path = f"{args.imitation_data_save_path}/{i.split('/')[-1]}"
            data = np.load(i)  # (600, 9)
            chunked_data = make_action_chunks(data, chunk_size=args.action_chunk_num)
            # print(chunked_data.shape)  # (600, 16, 9)
            np.save(save_path, chunked_data)
        
        print("===== 작업완료 =====")
