# -*- coding: utf-8 -*-
"""下载 CODE 最终图 (AIUB), 生成 data/gim_YYYY.npz。

    python code/download_gim_global.py 2015 2016 ... 2026
"""
import gzip
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np

import ionex_io as d
import paths as P

BASE = "https://www.aiub.unibe.ch/download/CODE/%d/"
HOSTS = [BASE, "https://download.aiub.unibe.ch/CODE/%d/"]
RAW = P.RAW


def names_for(year, doy, kind):
    yy = year % 100
    long_ok = (year, doy) >= (2022, 331)
    if kind == "fin":
        new = "COD0OPSFIN_%d%03d0000_01D_01H_GIM.INX.gz" % (year, doy)
        old = "CODG%03d0.%02dI.Z" % (doy, yy)
    else:
        # CODE 1 天预报: 短名 c1pg{ddd}0.{yy}i.Z，长名 COD0OPSP0D_...（IGSMAIL-8739；P1D 是 2 天预报，不要用）
        new = "COD0OPSP0D_%d%03d0000_01D_01H_GIM.INX.gz" % (year, doy)
        olds = ["c1pg%03d0.%02di.Z" % (doy, yy), "C1PG%03d0.%02dI.Z" % (doy, yy)]
        return [new] + olds if long_ok else olds + [new]
    return [new, old] if long_ok else [old, new]


def fetch_any(args):
    year, doy, kind = args
    hosts = [BASE] if kind == "fin" else HOSTS
    for n in names_for(year, doy, kind):
        dest = os.path.join(RAW, kind, n)
        for h in hosts:
            if d.fetch(h % year + n, dest, 1):
                return dest
    return None


def read_text(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", errors="ignore") as f:
            return f.read()
    import unlzw3
    with open(path, "rb") as f:
        return unlzw3.unlzw(f.read()).decode("ascii", "ignore")


def parse_one(args):
    path, year, doy = args
    try:
        ep, la, lo, tec = d.parse_ionex(read_text(path))
        day0 = datetime(year, 1, 1) + timedelta(days=doy - 1)
        keep = [k for k, e in enumerate(ep) if e.date() == day0.date()]
        hrs = [int(round((ep[k] - datetime(year, 1, 1)).total_seconds() / 3600.0)) for k in keep]
        a = tec[keep]
        if a.shape[1:] != (71, 73) or len(hrs) == 0:
            return doy, None, "网格 %s" % (a.shape,)
        if len(hrs) < 24:
            return doy, None, "当天只有 %d 张图 (不是 1 小时分辨率?)" % len(hrs)
        out = np.where(np.isfinite(a), np.rint(a * 10.0), -1).astype(np.int16)
        return doy, (hrs, out), ""
    except Exception as e:  # noqa
        return doy, None, repr(e)[:80]


def one_year(year, kind, pool):
    nd = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
    last = min(nd, (datetime.utcnow() - datetime(year, 1, 1)).days)
    if last <= 0:
        return
    os.makedirs(os.path.join(RAW, kind), exist_ok=True)
    if kind == "c1pg":      # 先探测 3 天，服务器上没有就不浪费时间
        probe = [fetch_any((year, doy, kind)) for doy in (min(last, 40), min(last, 120), min(last, 200))]
        print("%d C1PG 探测: %s" % (year, [os.path.basename(p) if p else None for p in probe]))
        if not any(probe):
            print("%d C1PG: 3 个探测日都没下到。原因见上面每条\"失败\"后的报错（404=文件不存在；超时/连接错误=网络问题），"
                  "先据此判断，不要直接认定服务器没有存档。评测时将只与持续性等基线比较。" % year)
            return
    with ThreadPoolExecutor(12) as ex:
        paths = list(ex.map(fetch_any, [(year, doy, kind) for doy in range(1, last + 1)]))
    jobs = [(p, year, doy) for doy, p in enumerate(paths, 1) if p]
    res = pool.map(parse_one, jobs, chunksize=8)
    full = np.full((nd * 24, 71, 73), -1, dtype=np.int16)
    bad = []
    for doy, r, msg in res:
        if r is None:
            bad.append((doy, msg))
            continue
        hrs, out = r
        full[hrs] = out
    name = "gim_%d.npz" % year if kind == "fin" else "c1pg_%d.npz" % year
    os.makedirs(P.DATA, exist_ok=True)
    np.savez_compressed(P.data(name), tec=full, year=year)
    ok_days = int((full.reshape(nd, 24, -1).max(axis=(1, 2)) >= 0).sum())
    print("%d %s: 有效 %d/%d 天, 下载失败 %d 天, 解析失败 %d 天 %s" % (
        year, kind, ok_days, last, sum(p is None for p in paths), len(bad), bad[:3]))


if __name__ == "__main__":
    years = [int(x) for x in sys.argv[1:]] or list(range(2015, 2027))
    if any(y <= 2022 for y in years):
        try:
            import unlzw3  # noqa
        except ImportError:
            raise SystemExit("需要先安装: pip install unlzw3")
    with Pool(max(2, (os.cpu_count() or 4) // 2)) as pool:
        for y in years:
            one_year(y, "fin", pool)
        for y in years:
            if y in (2019, 2024, 2025):
                one_year(y, "c1pg", pool)
    print("全部完成。")
