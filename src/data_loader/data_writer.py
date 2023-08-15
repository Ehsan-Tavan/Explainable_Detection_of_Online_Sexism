# -*- coding: utf-8 -*-
# ==========================================================================

"""
    Explainable Detection of Online Sexism Project:
        data loader:
            writing data
"""

# ============================ Third Party libs ============================
import json
import pickle

import pandas as pd


# ==========================================================================


def write_json(data: dict, path: str) -> None:
    """
    write_json function is written for write in json files

    Args:
        data:
        path:

    Returns:

    """
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4, ensure_ascii=False)


def write_pickle(path: str, data: list or dict) -> None:
    """
    write_pickle function is written for write data in pickle file

    Args:
        path:
        data:

    Returns:

    """
    with open(path, "wb") as outfile:
        pickle.dump(data, outfile)


def write_text(data: list, path: str) -> None:
    """
    save_text function is written for write in text files

    Args:
        data:
        path:

    Returns:

    """
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(data))


def write_excel(path: str, dataframe: pd.DataFrame) -> None:
    """
    write_excel function is written for write in excel files
    Args:
        path:
        dataframe:

    Returns:

    """
    dataframe.to_excel(path, index=False)


def write_csv(path: str, dataframe: pd.DataFrame) -> None:
    """
    write_csv function is written for write in csv files

    Args:
        path:
        dataframe:

    Returns:

    """
    dataframe.to_csv(path, index=False)
