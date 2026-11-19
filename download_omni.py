# -*- coding: utf-8 -*-
"""下载 NASA OMNI2 逐小时太阳风数据 (只在 --sw 时用)。"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_%d.dat"

if __name__ == "__main__":
    years = [int(x) for x in sys.argv[1:]] or list(range(2014, 2027))
    import paths as P
    od = os.path.join(P.DATA, "omni")
    os.makedirs(od, exist_ok=True)
    for y in years:
        dest = os.path.join(od, "omni2_%d.dat" % y)
        try:
            req = urllib.request.Request(URL % y, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            print("%d: %d 行" % (y, data.count(b"\n")))
        except Exception as e:  # noqa
            print("%d: 失败 %r" % (y, e))
    print("完成。")
