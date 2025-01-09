#!/usr/bin/env python
# -*- encoding=utf8 -*-

import os


def format_path(path_str: str):
    path_str = path_str.rstrip('\\/').replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" '\"\n\u202a")


def get_parent_abs_path(path_str: str):
    return os.path.dirname(os.path.abspath(path_str))


def get_base_path():
    current_path = os.path.abspath(__file__)
    src_path = os.path.join(os.path.dirname(current_path), '..', '..', '..')
    return os.path.abspath(src_path)
