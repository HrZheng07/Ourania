# -*- coding: utf-8 -*-
"""IONEX 下载与读取。"""
import time
import urllib.request
from datetime import datetime

import numpy as np


def fetch(url, dest, tries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 1000:
                raise IOError("file too small")
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:  # noqa
            err = e
            time.sleep(2 * (k + 1))
    print("    失败:", url, "->", err)
    return False


def parse_ionex(text):
    """返回 (epochs[list of datetime], lats, lons, tec[nt, nlat, nlon] 单位 TECU)。"""
    lines = text.splitlines()
    exponent = -1
    i = 0
    while i < len(lines):
        lab = lines[i][60:].strip()
        if lab == "EXPONENT":
            exponent = int(lines[i][:6])
        if lab == "END OF HEADER":
            break
        i += 1
    epochs, maps = [], []
    lats_ref, lons_ref = None, None
    while i < len(lines):
        lab = lines[i][60:].strip()
        if lab == "START OF TEC MAP":
            i += 1
            v = lines[i][:36].split()
            ep = datetime(*[int(x) for x in v[:6]])
            i += 1
            rows, lats = [], []
            while lines[i][60:].strip() != "END OF TEC MAP":
                if lines[i][60:].strip() == "LAT/LON1/LON2/DLON/H":
                    s = lines[i]
                    lat, lon1, lon2, dlon = (float(s[2:8]), float(s[8:14]),
                                             float(s[14:20]), float(s[20:26]))
                    n = int(round((lon2 - lon1) / dlon)) + 1
                    vals = []
                    i += 1
                    while len(vals) < n:
                        s = lines[i].rstrip("\n")
                        vals += [int(s[j:j + 5]) for j in range(0, len(s), 5) if s[j:j + 5].strip()]
                        i += 1
                    rows.append(vals[:n])
                    lats.append(lat)
                    lons = lon1 + dlon * np.arange(n)
                else:
                    i += 1
            epochs.append(ep)
            maps.append(rows)
            lats_ref, lons_ref = np.array(lats), lons
        elif lab in ("START OF RMS MAP", "END OF FILE"):
            break
        i += 1
    tec = np.array(maps, dtype=np.float64)
    tec[tec == 9999] = np.nan
    return epochs, lats_ref, lons_ref, tec * 10.0 ** exponent
