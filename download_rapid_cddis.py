# -*- coding: utf-8 -*-
"""下载 CODE 快速图 (CDDIS), 生成 data/rap_YYYY.npz。需要 NASA Earthdata 账号。

    python code/download_rapid_cddis.py 2015 2016 ... 2026
"""
import getpass
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np

import download_c1pg_cddis as base
import download_gim_global as g

import paths as P
HERE = P.DATA
base.RAW = os.path.join(P.RAW, "rap_cddis")


def names(year, doy):
    long_name = "COD0OPSRAP_%d%03d0000_01D_01H_GIM.INX.gz" % (year, doy)
    short = "corg%03d0.%02di.Z" % (doy, year % 100)
    return [long_name, short] if (year, doy) >= (2022, 331) else [short, long_name]


base.names = names            # base.fetch 通过模块全局变量 names / RAW 找文件, 这里替换成快速图


def parse_hourly(path, year, doy):
    """解析为逐小时图; 2 小时分辨率的文件用当天相邻图线性插值。"""
    _, r, msg = g.parse_one((path, year, doy))
    if r is not None:
        return r[0], r[1], False
    try:
        ep, la, lo, tec = g.d.parse_ionex(g.read_text(path))
        day0 = datetime(year, 1, 1) + timedelta(days=doy - 1)
        h = np.array([(e - day0).total_seconds() / 3600.0 for e in ep])
        sel = (h >= 0) & (h <= 24) & (np.abs(h - np.round(h)) < 1e-6)
        h, tec = np.round(h[sel]).astype(int), tec[sel]
        if tec.shape[1:] != (71, 73) or list(h) != list(range(0, 25, 2)):
            return None, msg, False
        out = np.empty((24, 71, 73))
        out[0::2] = tec[:12]
        out[1::2] = 0.5 * (tec[:12] + tec[1:13])
        hrs = [(day0 - datetime(year, 1, 1)).days * 24 + k for k in range(24)]
        return hrs, np.where(np.isfinite(out), np.rint(out * 10.0), -1).astype(np.int16), True
    except Exception as e:  # noqa
        return None, repr(e)[:80], False


def one_year(year):
    nd = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
    last = min(nd, (datetime.utcnow() - datetime(year, 1, 1)).days)
    probe, why = base.fetch(year, min(120, last))
    if probe is None:
        print("%d: 探测第 %d 天失败: %s" % (year, min(120, last), why))
        if why.startswith("HTTP 401") or "不是压缩文件" in why:
            raise SystemExit("账号认证失败，请检查用户名密码，或先在浏览器登录 https://cddis.nasa.gov/archive/ 授权后再试。")
        if why.startswith("HTTP 404"):
            print("   文件名可能不对。请在浏览器打开 https://cddis.nasa.gov/archive/gnss/products/ionex/%d/120/ "
                  "看看 CODE 快速图实际叫什么名字，告诉我。" % year)
        return
    with ThreadPoolExecutor(4) as ex:                 # CDDIS 限流，线程不要太多
        res = list(ex.map(lambda d: base.fetch(year, d), range(1, last + 1)))
    full = np.full((nd * 24, 71, 73), -1, dtype=np.int16)
    issued, fail, bad = {}, [], []
    n_interp = 0
    for doy, (p, why) in enumerate(res, 1):
        if p is None:
            fail.append((doy, why))
            continue
        hrs, out, interp = parse_hourly(p, year, doy)
        if hrs is None:
            bad.append((doy, out))
            continue
        full[hrs] = out
        if interp:
            n_interp += 1
        issued[doy] = base.issue_time(p)
    np.savez_compressed(os.path.join(HERE, "rap_%d.npz" % year), tec=full, year=year)
    with open(os.path.join(HERE, "rap_%d_issue.json" % year), "w", encoding="utf-8") as f:
        json.dump(dict(issued=issued, download_failed=fail, parse_failed=bad, days_interpolated_from_2h=n_interp), f, ensure_ascii=False, indent=1)
    print("%d 快速图: 有效 %d/%d 天 (其中 %d 天由 2 小时图插值), 下载失败 %d 天, 解析失败 %d 天" % (year, last - len(fail) - len(bad), last, n_interp, len(fail), len(bad)))
    if fail:
        print("   下载失败示例:", fail[:5])
    for d in (1, 120, 300):
        if d in issued:
            print("   第 %d 天的快速图生成时间: %s（应当是第 %d 天的次日）" % (d, issued[d], d))


if __name__ == "__main__":
    years = [int(x) for x in sys.argv[1:]] or [2024]
    if any(y < 2023 for y in years):
        try:
            import unlzw3  # noqa
        except ImportError:
            raise SystemExit("需要先安装: pip install unlzw3")
    user = input("Earthdata 用户名: ").strip()
    pwd = getpass.getpass("Earthdata 密码（输入时不显示）: ")
    base.AUTH = (user, pwd)
    for y in years:
        one_year(y)
    print("全部完成。下一步: python code/evaluate.py --ckpt ckpt_v3_ema_dp10.pt --input fin rap")
