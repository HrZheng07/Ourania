# -*- coding: utf-8 -*-
"""下载 CODE 1 天/2 天预报 C1PG/C2PG (CDDIS), 生成 data/c1pg_YYYY.npz。需要 NASA Earthdata 账号。

    python code/download_c1pg_cddis.py 2024 2025
    python code/download_c1pg_cddis.py 2026 c1pg c2pg
"""
import getpass
import gzip
import json
import threading
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np

import download_gim_global as g

import paths as P
HERE = P.DATA
RAW = os.path.join(P.RAW, "c1pg_cddis")
PROD = "c1pg"          # c1pg = 1 天预报; c2pg = 2 天预报 (提前 2 天发布, 文件名 COD0OPSP1D / c2pg)
URL = "https://cddis.nasa.gov/archive/gnss/products/ionex/%d/%03d/%s"
AUTH = None
_local = threading.local()

import requests


class EarthdataSession(requests.Session):
    """跳转到 urs.earthdata.nasa.gov 时保留认证信息。"""
    AUTH_HOST = "urs.earthdata.nasa.gov"

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        if "Authorization" in headers:
            a = requests.utils.urlparse(response.request.url).hostname
            b = requests.utils.urlparse(prepared_request.url).hostname
            if a != b and b != self.AUTH_HOST and a != self.AUTH_HOST:
                del headers["Authorization"]


def session():
    if not hasattr(_local, "s"):
        s = EarthdataSession()
        s.auth = AUTH
        s.headers["User-Agent"] = "Mozilla/5.0"
        _local.s = s
    return _local.s


def names(year, doy):
    tag = "P0D" if PROD == "c1pg" else "P1D"
    long_name = "COD0OPS%s_%d%03d0000_01D_01H_GIM.INX.gz" % (tag, year, doy)
    short = "%s%03d0.%02di.Z" % (PROD, doy, year % 100)
    return [long_name, short] if (year, doy) >= (2022, 331) else [short, long_name]


def fetch(year, doy):
    """返回 (本地路径 或 None, 失败原因)。"""
    err = ""
    for n in names(year, doy):
        dest = os.path.join(RAW if PROD == "c1pg" else RAW.replace("c1pg", PROD), str(year), n)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            return dest, ""
        for k in range(3):
            try:
                r = session().get(URL % (year, doy, n), timeout=90)
                if r.status_code != 200:
                    err = "HTTP %d" % r.status_code
                    if r.status_code in (401, 403, 404):
                        break
                    continue
                data = r.content
                if data[:2] not in (b"\x1f\x8b", b"\x1f\x9d"):        # 不是 gzip / compress，多半是登录页
                    with open(os.path.join(P.RAW, "cddis_debug.html"), "wb") as f:
                        f.write(data[:5000])
                    err = "返回的不是压缩文件（最终网址 %s，内容已存到 cddis_debug.html）" % r.url[:80]
                    break
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest + ".part", "wb") as f:
                    f.write(data)
                os.replace(dest + ".part", dest)
                return dest, ""
            except Exception as e:  # noqa
                err = repr(e)[:100]
                time.sleep(3 * (k + 1))
    return None, err


def parse_by_epoch(path, year):
    """按每张图的实际时刻放到该年的小时轴上。"""
    try:
        ep, la, lo, tec = g.d.parse_ionex(g.read_text(path))
        if tec.shape[1:] != (71, 73):
            return None, "网格 %s" % (tec.shape[1:],)
        t0 = datetime(year, 1, 1)
        h = np.array([(e - t0).total_seconds() / 3600.0 for e in ep])
        nd = (datetime(year + 1, 1, 1) - t0).days
        k = np.where((np.abs(h - np.round(h)) < 1e-6) & (h >= 0) & (h < nd * 24))[0]
        if True:                 # 文件覆盖的整天 (24 张) 都取, 次日 00 时那张丢掉
            days, cnt = np.unique(np.round(h[k]) // 24, return_counts=True)
            k = k[np.isin(np.round(h[k]) // 24, days[cnt >= 24])]
        if len(k) < 24:
            return None, "只有 %d 张整点图" % len(k)
        out = np.where(np.isfinite(tec[k]), np.rint(tec[k] * 10.0), -1).astype(np.int16)
        return ([int(round(x)) for x in h[k]], out), ""
    except Exception as e:  # noqa
        return None, repr(e)[:80]


def issue_time(path):
    """读 IONEX 头里的生成时间（PGM / RUN BY / DATE 行）。"""
    try:
        txt = g.read_text(path)
    except Exception:
        return None
    for ln in txt.splitlines()[:40]:
        if "PGM / RUN BY / DATE" in ln:
            return ln[40:60].strip()
    return None


def one_year(year):
    nd = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
    probe, why = fetch(year, 120)
    if probe is None:
        print("%d: 探测第 120 天失败: %s" % (year, why))
        if why.startswith("HTTP 401") or "不是压缩文件" in why:
            raise SystemExit("账号认证失败，请检查用户名密码，或先在浏览器登录 https://cddis.nasa.gov/archive/ 授权后再试。")
        return
    with ThreadPoolExecutor(4) as ex:                 # CDDIS 限流，线程不要太多
        res = list(ex.map(lambda d: fetch(year, d), range(1, nd + 1)))
    full = np.full((nd * 24, 71, 73), -1, dtype=np.int16)
    issued, fail, bad = {}, [], []
    for doy, (p, why) in enumerate(res, 1):
        if p is None:
            fail.append((doy, why))
            continue
        if PROD == "c1pg":       # C1PG 沿用论文中验证过的处理 (按目录日截取当天 24 张)
            _, r, msg = g.parse_one((p, year, doy))
        else:
            r, msg = parse_by_epoch(p, year)
        if r is None:
            bad.append((doy, msg))
            continue
        hrs, out = r
        full[hrs] = out
        issued[doy] = issue_time(p)
    np.savez_compressed(os.path.join(HERE, "%s_%d.npz" % (PROD, year)), tec=full, year=year)
    with open(os.path.join(HERE, "%s_%d_issue.json" % (PROD, year)), "w", encoding="utf-8") as f:
        json.dump(dict(issued=issued, download_failed=fail, parse_failed=bad), f, ensure_ascii=False, indent=1)
    print("%d %s: 有效 %d/%d 天, 下载失败 %d 天 (今年未来的日子当然没有), 解析失败 %d 天" % (year, PROD.upper(), nd - len(fail) - len(bad), nd, len(fail), len(bad)))
    if fail:
        print("   下载失败示例:", fail[:5])
    ex_days = [d for d in (1, 120, 300) if d in issued]
    for d in ex_days:
        print("   第 %d 天的预报文件生成时间: %s" % (d, issued[d]))


if __name__ == "__main__":
    prods = [x for x in sys.argv[1:] if x in ("c1pg", "c2pg")] or ["c1pg"]
    years = [int(x) for x in sys.argv[1:] if x.isdigit()] or [2024, 2025]
    if any(y < 2023 for y in years):
        try:
            import unlzw3  # noqa
        except ImportError:
            raise SystemExit("需要先安装: pip install unlzw3")
    user = input("Earthdata 用户名: ").strip()
    pwd = getpass.getpass("Earthdata 密码（输入时不显示）: ")
    AUTH = (user, pwd)
    for PROD in prods:
        for y in years:
            one_year(y)
    print("全部完成。")
