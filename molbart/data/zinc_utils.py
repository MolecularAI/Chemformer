from pathlib import Path

import numpy as np
import pandas as pd


def number_of_mols(data_path):
    path = Path(data_path)

    idx_file_mapping = []
    if path.is_dir():
        num_lines = 0
        for f in path.iterdir():
            text = f.read_text()
            num_mols = len(text.split("\n")) - 1
            idx_file_mapping.append((num_lines, num_lines + num_mols, f))
            num_lines += num_mols

    else:
        text = path.read_text()
        num_lines = len(text.split("\n"))
        idx_file_mapping.append((0, num_lines, path))

    return num_lines, idx_file_mapping


def read_df_slice(idxs, idx_file_mapping):
    """Read a slice of the dataset from disk by looking up the required files in the mapping

    Args:
        idxs (List[int]): Contiguous list of indices into the full dataset of molecules to read
        idx_file_mapping (dict): Mapping returned by number_of_mols function

    Returns:
        (pd.DataFrame): DataFrame of lines from dataset
    """

    file_idx_map = {}

    curr_idx = 0
    for start, end, file_path in idx_file_mapping:
        while curr_idx < len(idxs) and start <= idxs[curr_idx] < end:
            file_idx_map.setdefault(str(file_path), [])
            file_idx_map[str(file_path)].append(idxs[curr_idx] - start)
            curr_idx += 1

    dfs = []
    for file_path, file_idxs in file_idx_map.items():
        file_df = pd.read_csv(Path(file_path))
        df = file_df.iloc[file_idxs]
        dfs.append(df)

    df_slice = pd.concat(dfs, ignore_index=True, copy=False)
    return df_slice


def read_zinc_slice(data_path, rank, num_gpus, batch_size):
    num_mols, idx_file_mapping = number_of_mols(data_path)
    rank_idxs = [idxs.tolist() for idxs in np.array_split(list(range(num_mols)), num_gpus)]

    # Drop last mols to ensure all processes have the same number of batches
    num_mols = min([len(idxs) for idxs in rank_idxs])
    num_mols = batch_size * (num_mols // batch_size)
    idxs = rank_idxs[rank][:num_mols]

    df_slice = read_df_slice(idxs, idx_file_mapping)
    print(f"Read {str(len(df_slice.index))} molecules for gpu {str(rank)}")
    # How this df is utilised needs to be determined
    # dataset = ZincSlice(df_slice)
    return dataset
