# -*- coding: utf-8 -*-
"""下载 GFZ 的 Kp/ap/SN/F10.7 指数。不需要账号。"""
import os
import sys

import ionex_io as d
import paths as P

URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"


def main():
    os.makedirs(P.DATA, exist_ok=True)
    dest = P.data(os.path.basename(URL))
    if os.path.exists(dest) and "--redo" not in sys.argv:
        print("已存在:", dest, "(加 --redo 重新下载)")
        return
    if os.path.exists(dest):
        os.replace(dest, dest + ".old")
    print("下载", URL)
    print("完成" if d.fetch(URL, dest) else "失败, 请检查网络")


if __name__ == "__main__":
    main()
