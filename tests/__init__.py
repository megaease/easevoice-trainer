#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os
import sys


def append_src_to_path():
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"path for src: {path}")
    sys.path.append(path)
