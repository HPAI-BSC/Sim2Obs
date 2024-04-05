import os
import pickle
import shutil
from abc import ABC
from enum import Enum
from glob import glob
from pathlib import Path
from typing import AnyStr, Dict, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IOMode(Enum):
    BYTES = (0,)
    TEXT = (1,)
    IMG = 2


class IO(ABC):
    sep = os.sep

    def remove_extension(self, path):
        return Path(path).stem

    def path_join(self, a: AnyStr, *paths: AnyStr) -> str:
        return os.path.join(a, *paths)

    def get_basename(self, path: str) -> str:
        return os.path.basename(path)

    def exists_directory(self, path: str) -> bool:
        return os.path.exists(path)

    def exists_file(self, path: str) -> bool:
        return os.path.exists(path)

    def make_directory(self, path: str, overwrite: bool) -> None:
        if os.path.exists(path):
            if not overwrite:
                raise FileExistsError(
                    "Path already exists. Set `overwrite` to True to perform this action anyway"
                )
            shutil.rmtree(path)
        os.makedirs(path)

    def remove_directory(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)

    def remove_file(self, path: str) -> None:
        os.remove(path)

    def list_folders(self, path: str) -> List[str]:
        paths = glob(os.path.join(path, "*"))
        paths.sort()
        return [path + os.path.sep for path in paths if os.path.isdir(path)]

    def list_files(self, path: str, relpath_start: str = None) -> List[str]:
        files = glob(os.path.join(path, "*"))
        files.sort()
        if relpath_start:
            files = [os.path.relpath(file_, relpath_start) for file_ in files]
        return files

    ########
    # LOAD #
    ########

    def _load_bytes(self, path: str) -> object:
        with open(path, mode="rb") as f:
            return pickle.load(f)

    def _load_file(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path)
        return np.array(img)

    def load_data(self, path: str, mode: IOMode = IOMode.BYTES) -> Union[object, str, np.ndarray]:
        if mode == IOMode.IMG:
            return self._load_image(path)
        if mode == IOMode.TEXT:
            return self._load_file(path)
        if mode == IOMode.BYTES:
            return self._load_bytes(path)

    ########
    # DUMP #
    ########

    def dump_bytes(self, path: str, data: object) -> None:
        with open(path, mode="wb") as f:
            pickle.dump(data, f)

    def dump_file(self, path: str, data: str) -> None:
        with open(path, mode="w") as f:
            try:
                f.write(data)
            except UnicodeEncodeError:
                data = data.encode("ascii", "ignore").decode("ascii")
                f.write(data)

    def dump_image(self, path: str, data: np.ndarray) -> None:
        img = Image.fromarray(data)
        img.save(path)
