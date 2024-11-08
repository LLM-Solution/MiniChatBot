#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-18 12:35:20
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-08 08:25:20

""" Make full data script. """

# Built-in packages
from json import loads, dumps
from pathlib import Path
from random import seed, shuffle

# Third party packages

# Local packages

__all__ = []


if __name__ == "__main__":
    full_data = []
    DATA_PATH = Path("./data")
    for path in DATA_PATH.iterdir():
        print(f"Load data from {path}")
        # Load data
        with path.open("r") as f:
            full_data += loads(f.read())

    print("Shuffle data")
    seed(42)
    shuffle(full_data)

    path = DATA_PATH / 'full_data.json'
    print(f"Save full data {path}")
    # Save data
    with path.open("w") as f:
        f.write(dumps(full_data))
