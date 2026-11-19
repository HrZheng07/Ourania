# -*- coding: utf-8 -*-
"""目录约定: code/  data/  data/raw/  checkpoints/  results/"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
CKPT = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")
TRAIN = os.path.join(RESULTS, "training")
EVAL = os.path.join(RESULTS, "evaluation")


def data(fn):
    return os.path.join(DATA, fn)


def ckpt(fn):
    """检查点: 给文件名时在 checkpoints/ 里找; 给路径时原样使用。"""
    return fn if os.path.dirname(fn) else os.path.join(CKPT, fn)
