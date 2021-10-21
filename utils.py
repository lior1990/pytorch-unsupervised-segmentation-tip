import numpy as np


def replace_indices(arr: "np.array", n_channel, labels_start_index=0) -> "np.array":
    d = {}

    new_arr = np.zeros_like(arr)
    values = np.arange(n_channel) + labels_start_index
    free_index = 0

    for i, val in enumerate(arr):
        if val not in d:
            d[val] = values[free_index]
            free_index += 1

        new_arr[i] = d[val]

    return new_arr
