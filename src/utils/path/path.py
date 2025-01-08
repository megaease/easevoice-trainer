#!/usr/bin/env python
# -*- encoding=utf8 -*-

import os


def format_path(path_str: str):
    path_str = path_str.rstrip('\\/').replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" '\"\n\u202a")
