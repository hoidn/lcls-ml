import numpy as np

def summarize_structure(d, parent_key=''):
    """
    Recursively navigates through a nested dictionary and summarizes the structure,
    including names of keys and the data type, rank, and shape of each numpy array.

    Args:
    - d (dict): The nested dictionary to summarize.
    - parent_key (str): The parent key path for nested dictionaries.

    Returns:
    - List of tuples containing the key path, data type, and shape of numpy arrays.
    """
    summary = []
    for k, v in d.items():
        # Construct a new key path for nested dictionaries
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            # Recurse if the value is another dictionary
            summary.extend(summarize_structure(v, new_key))
        elif isinstance(v, np.ndarray):
            # Summarize numpy array: include its path, data type, and shape
            summary.append((new_key, v.dtype, v.shape))
    return summary
