from os import makedirs, path
from shutil import rmtree

DATA_DIR = './data'


def get_path(local_path, clear=False):
    p = path.join(DATA_DIR, local_path)
    if clear:
        rmtree(p, ignore_errors=True)
    makedirs(path.dirname(p), exist_ok=True)
    return p
