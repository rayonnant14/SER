import os


def check_if_exist(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
