#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-18 12:35:20
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-18 12:49:46

""" Make full data script. """

# Built-in packages
from json import loads, dumps
from pathlib import Path

# Third party packages

# Local packages

__all__ = []


if __name__ == "__main__":
    full_data = []
    DATA_PATH = Path("./data")
    for path in DATA_PATH.iterdir():
        print(f"Load {path}")
        # Load data
        with path.open("r") as f:
            full_data += loads(f.read())

    print("Save full data")
    # Save data
    with (DATA_PATH / "full_data.json").open("w") as f:
        f.write(dumps(full_data))