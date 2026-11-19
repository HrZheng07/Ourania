# -*- coding: utf-8 -*-
"""Ourania: 全球 TEC 图 24 小时预报模型 (训练与测试)。

输入过去 48 h 的全球 TEC 图和 17 个太阳/地磁驱动量, 输出未来 24 h 的图。
数据划分: 训练 2015-2022, 验证 2023, 测试 2024, 保留 2025 (--final)。

用法:
    python code/ourania.py --name ema_dp10 --amp                     # 主模型
    python code/ourania.py --name rapin --input-product rap --amp    # 快速版 (输入用快速图)
    python code/ourania.py --name x --dry                            # 只检查数据
"""
import argparse
import json
import math
import os
import sys
import time
import zipfile
from datetime import date, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths as P  # noqa: E402
Y0 = 2014                                   # 时间轴起点 2014-01-01 00UT（2014 年只用于指数的滑动平均）
SPLITS = {
    "chrono": ([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022], [2023], [2024, 2025]),
    "cross": ([2015, 2016, 2017, 2018, 2020, 2021, 2023, 2025], [2022], [2019, 2024]),
}
T_IN, T_OUT, HIST = 48, 24, 72
SCALE = 50.0
H = W = 72
LAT = np.concatenate([87.5 - 2.5 * np.arange(71), [-87.5]])     # 第 72 行是为了凑成 72 而复制的最后一行
LON = -180.0 + 5.0 * np.arange(72)
POLE_LAT, POLE_LON = 80.8, -72.7
NDRV = 17
LOGS = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    sys.stdout.flush()
    LOGS.append(s)


def day_index(y, m=1, dd=1):
    return (date(y, m, dd) - date(Y0, 1, 1)).days


# ----------------------------------------------------------------------------
# 数据 (numpy)
# ----------------------------------------------------------------------------
def load_maps(prefix, years):
    """返回 (arr[int16, N,72,72], 可用年份); 缺失为 -1。"""
    have = [y for y in years if os.path.exists(P.data("%s_%d.npz" % (prefix, y)))]
    if not have:
        return None, []
    n = day_index(max(have) + 1) * 24
    arr = np.full((n, H, W), -1, dtype=np.int16)
    for y in have:
        t = np.load(P.data("%s_%d.npz" % (prefix, y)))["tec"]
        o = day_index(y) * 24
        arr[o:o + t.shape[0], :71] = t[:, :, :72]
        arr[o:o + t.shape[0], 71] = t[:, 70, :72]
    return arr, have


def load_indices(n_days):
    f107 = np.full(n_days, np.nan)
    sn = np.full(n_days, np.nan)
    ap = np.full(n_days * 8, np.nan)
    kp = np.full(n_days * 8, np.nan)
    p = P.data("Kp_ap_Ap_SN_F107_since_1932.txt")
    if not os.path.exists(p):
        raise SystemExit("缺少 Kp_ap_Ap_SN_F107_since_1932.txt，先运行 python code/download_indices.py")
    with open(p, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            q = line.split()
            try:
                y = int(q[0])
                if y < Y0:
                    continue
                di = day_index(y, int(q[1]), int(q[2]))
                if di >= n_days:
                    continue
                kp[di * 8:di * 8 + 8] = [float(v) for v in q[7:15]]
                ap[di * 8:di * 8 + 8] = [float(v) for v in q[15:23]]
                sn[di] = float(q[24])
                f107[di] = float(q[25])
            except Exception:
                continue
    f107[f107 <= 0] = np.nan
    sn[sn < 0] = np.nan
    ap[ap < 0] = np.nan
    kp[kp < 0] = np.nan

    def fill(a, default, name):
        """因果填补: 缺失处用此前最后一个有效值。"""
        g = np.isfinite(a)
        last = np.where(g, np.arange(len(a)), -1)
        np.maximum.accumulate(last, out=last)
        out = np.where(last >= 0, a[np.maximum(last, 0)], default)
        n_mid, n_head = int((~g & (last >= 0)).sum()), int((last < 0).sum())
        if n_mid or n_head:
            log("指数 %s: 用前值填补 %d 个, 开头用默认值 %s 填补 %d 个" % (name, n_mid, default, n_head))
        return out
    f107, sn, ap = fill(f107, 100.0, "F10.7"), fill(sn, 50.0, "SN"), fill(ap, 7.0, "ap")
    c = np.concatenate([[0.0], np.cumsum(f107)])
    i = np.arange(n_days)
    lo = np.maximum(0, i - 80)
    f81 = (c[i + 1] - c[lo]) / (i + 1 - lo)        # 截止到当天的 81 天滑动平均
    return dict(f107=f107, f81=f81, sn=sn, ap=ap, kp=kp)


def driver_table(idx, n_hours):
    """每个预报起点 t0 的驱动量向量 (只用 t0 之前的信息)。"""
    t0 = np.arange(n_hours)
    d1 = np.clip((t0 - 24) // 24, 0, None)
    d2 = np.clip((t0 - 48) // 24, 0, None)
    k0 = t0 // 3
    cols = [idx["f107"][d1] / 150.0, idx["f107"][d2] / 150.0, idx["f81"][d1] / 150.0, idx["sn"][d1] / 150.0]
    aps = []
    for j in range(1, 9):
        a = idx["ap"][np.clip(k0 - j, 0, None)]
        aps.append(a)
        cols.append(np.log1p(a) / 4.0)
    cols.append(np.log1p(np.mean(aps, axis=0)) / 4.0)
    doy = np.array([(date(Y0, 1, 1) + timedelta(days=int(x))).timetuple().tm_yday for x in range(n_hours // 24 + 1)])
    ang = 2 * math.pi * doy[t0 // 24] / 365.25
    ut = 2 * math.pi * (t0 % 24) / 24.0
    cols += [np.sin(ang), np.cos(ang), np.sin(ut), np.cos(ut)]
    tab = np.stack(cols, axis=1).astype(np.float32)
    assert tab.shape[1] == NDRV
    kpmax = np.array([np.nanmax(idx["kp"][k:k + 8]) if k + 8 <= len(idx["kp"]) and np.isfinite(idx["kp"][k:k + 8]).any()
                      else np.nan for k in k0])
    return tab, doy[t0 // 24].astype(np.float32), kpmax


# ---- 太阳风 / 地磁驱动量 (NASA OMNI2 逐小时, --sw 时使用) ----
OMNI_COLS = dict(bz=(16, 999.9), n=(23, 999.9), v=(24, 9999.0), p=(28, 99.99), e=(35, 999.99), dst=(40, 99999.0), ae=(41, 9999.0))


def load_omni(n_hours):
    """读 OMNI2, 返回逐小时数组; 缺失用此前的值填补。"""
    out = {k: np.full(n_hours, np.nan) for k in OMNI_COLS}
    files = sorted(os.listdir(os.path.join(P.DATA, "omni"))) if os.path.isdir(os.path.join(P.DATA, "omni")) else []
    if not files:
        raise SystemExit("缺少 omni/omni2_YYYY.dat，请先运行 python download_omni.py")
    for fn in files:
        if not fn.startswith("omni2_"):
            continue
        for ln in open(os.path.join(P.DATA, "omni", fn), errors="ignore"):
            q = ln.split()
            if len(q) < 42:
                continue
            y, dd, hh = int(q[0]), int(q[1]), int(q[2])
            if y < Y0:
                continue
            t = day_index(y) * 24 + (dd - 1) * 24 + hh
            if not 0 <= t < n_hours:
                continue
            for k, (c, fill) in OMNI_COLS.items():
                v = float(q[c])
                if abs(v) < fill * 0.999:
                    out[k][t] = v
    for k, a in out.items():
        g = np.isfinite(a)
        last = np.where(g, np.arange(n_hours), -1)
        np.maximum.accumulate(last, out=last)
        n_mid, n_head = int((~g & (last >= 0)).sum()), int((last < 0).sum())
        out[k] = np.where(last >= 0, a[np.maximum(last, 0)], 0.0)
        log("OMNI %s: 用前值填补 %d 小时, 开头缺失 %d 小时 (置 0)" % (k, n_mid, n_head))
    return out


def sw_table(om, n_hours):
    """每个起报时刻 t0 的太阳风特征 (只用 t0-1 及更早的小时)。13 列。"""
    t0 = np.arange(n_hours)
    prev = np.clip(t0 - 1, 0, None)

    def mean_k(a, k):
        c = np.concatenate([[0.0], np.cumsum(a)])
        lo = np.clip(t0 - k, 0, None)
        return (c[t0] - c[lo]) / np.maximum(t0 - lo, 1)

    def win(a, k, f):
        pad = np.concatenate([np.full(k, a[0]), a])
        w = np.lib.stride_tricks.sliding_window_view(pad, k)          # w[i] = a[i-k .. i-1]
        return f(w[:n_hours], axis=1)

    bz, v, n, p, e, dst, ae = (om[k] for k in ("bz", "v", "n", "p", "e", "dst", "ae"))
    cols = [bz[prev] / 10.0, mean_k(bz, 6) / 10.0, win(bz, 24, np.min) / 10.0,
            (v[prev] - 400.0) / 200.0, (mean_k(v, 6) - 400.0) / 200.0,
            mean_k(n, 6) / 10.0, mean_k(p, 6) / 5.0, mean_k(e, 6) / 5.0,
            dst[prev] / 100.0, win(dst, 24, np.min) / 100.0, (dst[prev] - dst[np.clip(t0 - 7, 0, None)]) / 50.0,
            mean_k(ae, 6) / 500.0, win(ae, 24, np.max) / 500.0]
    return np.stack(cols, axis=1).astype(np.float32)


def valid_starts(arr):
    ok = (arr.reshape(arr.shape[0], -1).min(axis=1) >= 0).astype(np.int64)    # 每张图所有格点都有效
    c = np.concatenate([[0], np.cumsum(ok)])
    n = arr.shape[0]
    t0 = np.arange(HIST, n - T_OUT + 1)
    good = (c[t0 + T_OUT] - c[t0 - HIST]) == (HIST + T_OUT)
    return t0[good]


def split_starts(t0, years):
    out = []
    for y in years:
        a, b = day_index(y) * 24, day_index(y + 1) * 24
        out.append(t0[(t0 - HIST >= a) & (t0 + T_OUT <= b)])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def cos_zenith(xp, lat, lon, ut, doy):
    g = (2.0 * math.pi / 366.0) * (doy - 1.0)
    dec = (0.006918 - 0.399912 * xp.cos(g) + 0.070257 * xp.sin(g) - 0.006758 * xp.cos(2 * g)
           + 0.000907 * xp.sin(2 * g) - 0.002697 * xp.cos(3 * g) + 0.00148 * xp.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * xp.cos(g) - 0.032077 * xp.sin(g)
                    - 0.014615 * xp.cos(2 * g) - 0.040849 * xp.sin(2 * g))
    h = (ut + lon / 15.0 + eot / 60.0 - 12.0) * (math.pi / 12.0)
    la = lat * (math.pi / 180.0)
    return xp.sin(la) * xp.sin(dec) + xp.cos(la) * xp.cos(dec) * xp.cos(h)


def maglat_np(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    pl, po = math.radians(POLE_LAT), math.radians(POLE_LON)
    return np.degrees(np.arcsin(np.sin(la) * math.sin(pl) + np.cos(la) * math.cos(pl) * np.cos(lo - po)))


# ----------------------------------------------------------------------------
# 模型 (PyTorch)
# ----------------------------------------------------------------------------
def build_model(torch, C=16, D=512, NB=8, use_drivers=True, drop_path=0.0, ndrv=NDRV, n_aux=2 + T_OUT, sunframe=False):
    nn, F = torch.nn, torch.nn.functional

    class PadConv(nn.Module):
        """经度方向循环填充、纬度方向复制填充的卷积。"""
        def __init__(self, cin, cout, k, stride=1, groups=1):
            super().__init__()
            self.p = k // 2
            self.conv = nn.Conv2d(cin, cout, k, stride, 0, groups=groups)

        def forward(self, x):
            if self.p:
                x = F.pad(x, (self.p, self.p, 0, 0), mode="circular")
                x = F.pad(x, (0, 0, self.p, self.p), mode="replicate")
            return self.conv(x)

    class Block(nn.Module):
        def __init__(self, dp=0.0):
            super().__init__()
            self.dp = dp
            self.norm = nn.GroupNorm(8, D)
            self.dw = PadConv(D, D, 7, groups=D)
            self.pw1 = nn.Conv2d(D, 2 * D, 1)
            self.pw2 = nn.Conv2d(2 * D, D, 1)
            self.film = nn.Linear(128, 2 * D)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        def forward(self, x, emb):
            h = self.norm(x)
            gb = self.film(emb)
            h = h * (1.0 + gb[:, :D, None, None]) + gb[:, D:, None, None]
            h = self.pw2(F.gelu(self.pw1(self.dw(h))))
            if self.training and self.dp > 0:          # 随机深度: 按样本随机跳过这个块
                keep = (torch.rand(h.shape[0], 1, 1, 1, device=h.device) >= self.dp).to(h.dtype)
                h = h * keep / (1.0 - self.dp)
            return x + h

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = PadConv(1, C, 3)
            self.enc2 = PadConv(C, C, 3, stride=2)
            self.enc3 = PadConv(C, C, 3, stride=2)
            self.inp = nn.Conv2d(T_IN * C + n_aux, D, 1)
            self.drv = nn.Sequential(nn.Linear(ndrv, 128), nn.SiLU(), nn.Linear(128, 128))
            self.blocks = nn.ModuleList([Block(drop_path * k / max(NB - 1, 1)) for k in range(NB)])
            self.out = nn.Conv2d(D, T_OUT * C, 1)
            self.dec1 = PadConv(C, C, 3)
            self.dec2 = PadConv(C, C, 3)
            self.dec3 = PadConv(2 * C, C, 3)
            self.head = nn.Conv2d(C, 1, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x, aux, drv, t0=None):
            if sunframe:                       # 输入转到太阳固定坐标, 输出再转回地理坐标
                ar_in = torch.arange(-T_IN, 0, device=x.device)
                ar_out = torch.arange(0, T_OUT, device=x.device)
                x = roll_lon(torch, x, sun_shift(torch, (t0[:, None] + ar_in[None]).float()))
                r = self.core(x, aux, drv)
                return roll_lon(torch, r, -sun_shift(torch, (t0[:, None] + ar_out[None]).float()))
            return self.core(x, aux, drv)

        def core(self, x, aux, drv):
            B = x.shape[0]
            e1 = F.silu(self.enc1(x.reshape(B * T_IN, 1, H, W)))
            e3 = F.silu(self.enc3(F.silu(self.enc2(e1))))
            z = e3.reshape(B, T_IN * C, H // 4, W // 4)
            z = self.inp(torch.cat([z, aux], dim=1))
            emb = self.drv(drv) if use_drivers else torch.zeros(B, 128, device=x.device, dtype=z.dtype)
            for blk in self.blocks:
                z = blk(z, emb)
            y = self.out(z).reshape(B * T_OUT, C, H // 4, W // 4)
            y = F.silu(self.dec1(F.interpolate(y, scale_factor=2, mode="nearest")))
            y = F.silu(self.dec2(F.interpolate(y, scale_factor=2, mode="nearest")))
            skip = e1.reshape(B, T_IN, C, H, W)[:, T_IN - T_OUT:].reshape(B * T_OUT, C, H, W)
            y = F.silu(self.dec3(torch.cat([y, skip], dim=1)))
            return self.head(y).reshape(B, T_OUT, H, W)
    return Net()


def build_convlstm(torch, hid=64, layers=2, C=32, use_drivers=False, ndrv=NDRV):
    """ConvLSTM 基线 (encoder-forecaster)。"""
    nn, F = torch.nn, torch.nn.functional

    class PadConv(nn.Module):
        def __init__(self, cin, cout, k, stride=1):
            super().__init__()
            self.p = k // 2
            self.conv = nn.Conv2d(cin, cout, k, stride, 0)

        def forward(self, x):
            if self.p:
                x = F.pad(x, (self.p, self.p, 0, 0), mode="circular")
                x = F.pad(x, (0, 0, self.p, self.p), mode="replicate")
            return self.conv(x)

    class Cell(nn.Module):
        def __init__(self, cin):
            super().__init__()
            self.g = PadConv(cin + hid, 4 * hid, 3)
            self.norm = nn.GroupNorm(8, 4 * hid)

        def forward(self, x, h, c):
            i, f, o, g = torch.chunk(self.norm(self.g(torch.cat([x, h], 1))), 4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, c

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(PadConv(1, C, 3), nn.SiLU(), PadConv(C, C, 3, stride=2), nn.SiLU())
            self.e_cells = nn.ModuleList([Cell(C if k == 0 else hid) for k in range(layers)])
            self.f_cells = nn.ModuleList([Cell(C if k == 0 else hid) for k in range(layers)])
            self.f_in = nn.Parameter(torch.zeros(1, C, H // 2, W // 2))
            self.drv = nn.Linear(ndrv, C) if use_drivers else None
            self.dec = nn.Sequential(PadConv(hid, C, 3), nn.SiLU())
            self.head = nn.Conv2d(C, 1, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x, aux, drv):
            B = x.shape[0]
            h = [torch.zeros(B, hid, H // 2, W // 2, device=x.device, dtype=x.dtype) for _ in range(layers)]
            c = [torch.zeros_like(h[0]) for _ in range(layers)]
            for t in range(T_IN):
                z = self.enc(x[:, t:t + 1])
                for k, cell in enumerate(self.e_cells):
                    h[k], c[k] = cell(z, h[k], c[k])
                    z = h[k]
            outs = []
            fin = self.f_in.expand(B, -1, -1, -1)
            if self.drv is not None:
                fin = fin + self.drv(drv)[:, :, None, None]
            for t in range(T_OUT):
                z = fin
                for k, cell in enumerate(self.f_cells):
                    h[k], c[k] = cell(z, h[k], c[k])
                    z = h[k]
                y = self.dec(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
                outs.append(self.head(y))
            return torch.cat(outs, 1)
    return Net()


def build_openstl_simvp(torch, hid_S=32, hid_T=256, N_S=4, N_T=8, model_type="gSTA"):
    """OpenSTL SimVPv2 基线: 输入 48 帧, 取前 24 帧输出。"""
    try:
        from openstl_local import SimVP_Model
    except ImportError:
        from openstl.models import SimVP_Model
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = SimVP_Model(in_shape=(T_IN, 1, H, W), hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T, model_type=model_type)

        def forward(self, x, aux, drv):
            return self.m(x[:, :, None])[:, :T_OUT, 0]
    return Net()


class Bundle:
    """把数据放到 GPU 上并按起点索引取批。"""
    def __init__(self, torch, dev, arr, drv, doy, use_solar, sunframe=False, inp=None):
        self.torch, self.dev = torch, dev
        self.data = torch.from_numpy(arr).to(dev)                 # int16, 0.1 TECU; 目标 (真值)
        self.inp = self.data if inp is None else torch.from_numpy(inp).to(dev)   # 输入图 (默认与目标同一产品)
        self.drv = torch.from_numpy(drv).to(dev)
        self.doy = torch.from_numpy(doy).to(dev)
        LA, LO = np.meshgrid(LAT, LON, indexing="ij")
        ml = maglat_np(LA, LO)
        pool = lambda a: a.reshape(H // 4, 4, W // 4, 4).mean(axis=(1, 3))
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        self.lat_s, self.lon_s = t(pool(LA)), t(pool(LO))
        self.static = torch.stack([t(pool(ml)) / 90.0, torch.sin(t(pool(LA)) * math.pi / 180.0)])   # [2,18,18]
        self.lat_f, self.lon_f, self.x_f = t(LA), t(LO), t(ml) / 90.0
        self.ar_in = torch.arange(-T_IN, 0, device=dev)
        self.ar_out = torch.arange(0, T_OUT, device=dev)
        self.use_solar = use_solar
        self.sunframe = sunframe

    def frames(self, t0, offs, src=None):
        src = self.data if src is None else src
        return src[t0[:, None] + offs[None, :]].float() / (10.0 * SCALE)

    def cz(self, t0, lat, lon, offs):
        tt = (t0[:, None] + offs[None, :]).float()
        ut = torch_mod(self.torch, tt, 24.0)[:, :, None, None]
        doy = (self.doy[t0][:, None] + (torch_mod(self.torch, t0.float(), 24.0)[:, None] + offs[None, :].float()) / 24.0)[:, :, None, None]
        return cos_zenith(self.torch, lat[None, None], lon[None, None], ut, doy)

    def n_aux(self):
        return 1 + 2 * T_OUT if self.sunframe else 2 + T_OUT

    def aux_sun(self, t0):
        """日固坐标下的辅助通道: sin(纬度)、地磁纬度、太阳天顶角余弦。"""
        B = t0.shape[0]
        sh = sun_shift(self.torch, (t0[:, None] + self.ar_out[None]).float())
        ml = roll_lon(self.torch, self.x_f[None, None].expand(B, T_OUT, H, W).contiguous(), sh)
        if self.use_solar:
            cz = roll_lon(self.torch, self.cz(t0, self.lat_f, self.lon_f, self.ar_out).contiguous(), sh)
        else:
            cz = self.torch.zeros(B, T_OUT, H, W, device=self.dev)
        pool = lambda a: self.torch.nn.functional.avg_pool2d(a, 4)
        return self.torch.cat([self.static[None, 1:2].expand(B, -1, -1, -1), pool(ml), pool(cz)], dim=1)

    def batch(self, t0):
        x = self.frames(t0, self.ar_in, self.inp)
        y = self.frames(t0, self.ar_out)
        B = t0.shape[0]
        if self.sunframe:
            return x, y, self.aux_sun(t0), self.drv[t0]
        cz = self.cz(t0, self.lat_s, self.lon_s, self.ar_out) if self.use_solar else \
            self.torch.zeros(B, T_OUT, H // 4, W // 4, device=self.dev)
        aux = self.torch.cat([self.static[None].expand(B, -1, -1, -1), cz], dim=1)
        return x, y, aux, self.drv[t0]


def torch_mod(torch, a, m):
    return a - m * torch.floor(a / m)


def sun_shift(torch, hours):
    """日固坐标: 世界时 u 的图沿经度平移 3u+36 格。"""
    return (3 * torch_mod(torch, hours, 24) + 36).long()


def roll_lon(torch, a, s):
    """每个样本每一帧沿经度循环平移 s 格。"""
    B, T, Hh, Ww = a.shape
    idx = torch_mod(torch, torch.arange(Ww, device=a.device)[None, None, None, :] - s[:, :, None, None], Ww).long()
    return a.gather(3, idx.expand(B, T, Hh, Ww))


W_LAT = np.cos(np.radians(LAT[:71]))
W_LAT = (W_LAT / W_LAT.mean()).astype(np.float32)        # 按纬度余弦的面积权重, 均值为 1


class Acc:
    def __init__(self):
        self.d = {}
        self.w = {}

    def addw(self, key, err, w):
        """面积加权平方误差, err 最后两维是 (71, 72) 的全球网格。"""
        s = self.w.setdefault(key, [0.0, 0])
        s[0] += float((err * err * w[:, None]).sum())
        s[1] += int(err.numel())

    def add(self, key, err):
        s = self.d.setdefault(key, [0.0, 0.0, 0])
        s[0] += float((err * err).sum())
        s[1] += float(err.abs().sum())
        s[2] += int(err.numel())

    def out(self):
        r = {k: dict(rmse=math.sqrt(v[0] / max(v[2], 1)), mae=v[1] / max(v[2], 1), n=v[2]) for k, v in self.d.items()}
        for k, v in self.w.items():
            r.setdefault(k, {})["rmse_w"] = math.sqrt(v[0] / max(v[1], 1))
        return r


def evaluate(torch, model, bd, starts, kpmax, c1pg, pde, a, save_days=None):
    """在给定起点上评测。误差单位 TECU，只统计真实的 71 行。"""
    model.eval()
    dev = bd.dev
    acc = Acc()
    lead = {k: torch.zeros(T_OUT, device=dev) for k in ("model", "pers", "c1pg", "model@c1pg")}
    lead_n = {k: 0 for k in lead}
    lat_ok = torch.tensor((LAT >= 15) & (LAT <= 55), device=dev)
    lon_ok = torch.tensor((LON >= 70) & (LON <= 140), device=dev)
    saved = {}
    wl = torch.tensor(W_LAT, device=dev)
    with torch.no_grad():
        for i in range(0, len(starts), a.batch):
            t0 = torch.as_tensor(starts[i:i + a.batch], device=dev)
            x, y, aux, drv = bd.batch(t0)
            pred = predict(torch, model, x, aux, drv, a, t0)
            T = lambda z: z[:, :, :71] * SCALE
            p, yt = T(pred), T(y)
            pers = T(x[:, T_IN - T_OUT:])
            m3 = T(sum(bd.frames(t0, bd.ar_out - 24 * k, bd.inp) for k in (1, 2, 3)) / 3.0)   # 与输入同一产品
            e = dict(model=p - yt, pers=pers - yt, mean3=m3 - yt)
            kp = kpmax[starts[i:i + a.batch]]
            storm = torch.tensor(np.nan_to_num(kp, nan=0.0) >= 5.0, device=dev)
            for k, v in e.items():
                acc.add(k, v)
                acc.addw(k, v, wl)
                acc.add(k + "|china", v[:, :, lat_ok[:71]][:, :, :, lon_ok])
                if storm.any():
                    acc.add(k + "|storm", v[storm])
                if (~storm).any():
                    acc.add(k + "|quiet", v[~storm])
            lead["model"] += (e["model"] ** 2).mean(dim=(2, 3)).sum(0)
            lead["pers"] += (e["pers"] ** 2).mean(dim=(2, 3)).sum(0)
            lead_n["model"] += len(t0)
            lead_n["pers"] += len(t0)
            if c1pg is not None:
                tt = (t0[:, None] + bd.ar_out[None, :]).cpu().numpy()
                inside = tt.max(axis=1) < c1pg.shape[0]
                cc = np.full(tt.shape + (H, W), -1, dtype=np.int16)
                cc[inside] = c1pg[tt[inside]]
                has = torch.tensor(cc.reshape(len(t0), -1).min(axis=1) >= 0, device=dev)
                if has.any():
                    cp = torch.from_numpy(cc).to(dev).float()[:, :, :71] / 10.0
                    ec = (cp - yt)[has]
                    acc.add("c1pg", ec)
                    acc.add("model@c1pg", e["model"][has])
                    acc.addw("c1pg", ec, wl)
                    acc.addw("model@c1pg", e["model"][has], wl)
                    acc.add("pers@c1pg", e["pers"][has])
                    acc.add("mean3@c1pg", e["mean3"][has])
                    acc.add("c1pg|china", ec[:, :, lat_ok[:71]][:, :, :, lon_ok])
                    acc.add("model@c1pg|china", e["model"][has][:, :, lat_ok[:71]][:, :, :, lon_ok])
                    if (storm & has).any():
                        acc.add("c1pg|storm", (cp - yt)[storm & has])
                        acc.add("model@c1pg|storm", e["model"][storm & has])
                    lead["c1pg"] += (ec ** 2).mean(dim=(2, 3)).sum(0)
                    lead["model@c1pg"] += (e["model"][has] ** 2).mean(dim=(2, 3)).sum(0)
                    lead_n["c1pg"] += int(has.sum())
                    lead_n["model@c1pg"] += int(has.sum())
            if save_days is not None:
                for j, s in enumerate(starts[i:i + a.batch]):
                    if int(s) in save_days:
                        saved["pred_%d" % s] = (p[j] * 10).round().clamp(0, 32000).short().cpu().numpy()
                        saved["true_%d" % s] = (yt[j] * 10).round().short().cpu().numpy()
    res = acc.out()
    res["rmse_by_lead"] = {k: [math.sqrt(x) for x in (v / max(lead_n[k], 1)).tolist()] for k, v in lead.items() if lead_n[k] > 0}
    res["n_starts"] = int(len(starts))
    model.train()
    return res, saved


def predict(torch, model, x, aux, drv, a, t0=None):
    r = model(x, aux, drv, t0) if getattr(a, "sunframe", False) else model(x, aux, drv)
    return (x[:, T_IN - T_OUT:] + r) if not a.no_residual else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry", action="store_true", help="只检查数据与样本划分，不训练")
    ap.add_argument("--no-drivers", action="store_true")
    ap.add_argument("--no-solar", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--lam-pde", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arch", default="v3", choices=["v3", "convlstm", "convlstm_drv", "openstl_simvp"],
                    help="模型结构: v3 (本文模型) / convlstm (纯 TEC 的编码-预报 ConvLSTM) / convlstm_drv (加驱动量) / openstl_simvp (OpenSTL 官方 SimVPv2)")
    ap.add_argument("--sunframe", action="store_true", help="在太阳固定坐标系里预报 (输入按世界时沿经度平移, 输出再转回), 只用于 --arch v3")
    ap.add_argument("--input-product", default="fin", choices=["fin", "rap"],
                    help="模型输入用哪种图: fin = CODE 最终图 (论文主模型); rap = CODE 快速图 (业务可得; 目标和真值仍是最终图)")
    ap.add_argument("--sw", action="store_true", help="加入 OMNI 太阳风/地磁驱动量 (Bz、速度、密度、动压、电场、Dst、AE; 只用起报前的值)")
    ap.add_argument("--loss", default="l1", choices=["l1", "mse", "l1mse"])
    ap.add_argument("--wd", type=float, default=0.01, help="AdamW 权重衰减")
    ap.add_argument("--drop-path", type=float, default=0.1, help="随机深度的最大跳过概率 (最后一个块), 例如 0.1")
    ap.add_argument("--eval-every", type=int, default=1000, help="每多少步在验证集上评测一次")
    ap.add_argument("--ema", type=float, default=0.999, help="权重滑动平均的衰减率, 例如 0.999 (0 = 不用); 评测和保存都用平均后的权重")
    ap.add_argument("--patience", type=int, default=8, help="验证 RMSE 连续多少次评测没有改进就停止 (0 = 不提前停止)")
    ap.add_argument("--split", default="chrono", choices=list(SPLITS))
    ap.add_argument("--eval-only", default="", help="只评测: 给出已训练模型的检查点文件名, 例如 ckpt_v3_main.pt")
    ap.add_argument("--final", action="store_true", help="评测保留的最终测试年 2025（模型定型后只跑一次）")
    a = ap.parse_args()
    if a.quick:
        a.name, a.iters = "quick", 60

    TRAIN_Y, VAL_Y, TEST_Y = SPLITS[a.split]
    all_years = sorted(set(TRAIN_Y + VAL_Y + TEST_Y + [2026]))
    arr, have = load_maps("gim", all_years)
    if arr is None:
        raise SystemExit("没有找到 gim_YYYY.npz，请先运行 download_gim_global.py")
    log("已有年份: %s, 时间轴 %d 小时" % (have, arr.shape[0]))
    idx = load_indices(arr.shape[0] // 24 + 1)
    drv, doy, kpmax = driver_table(idx, arr.shape[0])
    if a.sw:
        drv = np.concatenate([drv, sw_table(load_omni(arr.shape[0]), arr.shape[0])], axis=1)
        log("加入太阳风驱动量, 驱动量维数 %d" % drv.shape[1])
    t0_all = valid_starts(arr)
    arr_in = None
    if a.input_product == "rap":
        ri, rhave = load_maps("rap", all_years)
        if ri is None:
            raise SystemExit("没有 rap_YYYY.npz, 先运行 python code/download_rapid_cddis.py 2015 ... 2024")
        arr_in = np.full_like(arr, -1)
        n = min(arr.shape[0], ri.shape[0])
        arr_in[:n] = ri[:n]
        del ri
        okin = np.concatenate([[0], np.cumsum(arr_in.reshape(arr_in.shape[0], -1).min(axis=1) >= 0)])
        n0 = len(t0_all)
        t0_all = t0_all[(okin[t0_all] - okin[t0_all - HIST]) == HIST]    # 输入的 72 小时历史也要完整
        log("输入产品: CODE 快速图 (有 %s 年); 要求快速图历史完整后起点 %d -> %d" % (rhave, n0, len(t0_all)))
    if a.quick:
        tr = va = t0_all
        tests = {str(have[-1]): t0_all[t0_all % 24 == 0][:20]}
        va = va[va % 24 == 0][:20]
    else:
        tr = split_starts(t0_all, [y for y in TRAIN_Y if y in have])
        va = split_starts(t0_all, [y for y in VAL_Y if y in have])
        va = va[va % 24 == 0]
        tests = {}
        for y in TEST_Y:
            if a.split == "chrono" and y == 2025 and not a.final:
                log("2025 年是保留的最终测试年，本次不评测（模型定型后加 --final）")
                continue
            if y in have:
                s = split_starts(t0_all, [y])
                tests[str(y)] = s[s % 24 == 0]
        missing = [y for y in TRAIN_Y + VAL_Y + TEST_Y if y not in have]
        if missing:
            log("警告: 缺少年份 %s 的数据，结果不能与计划的划分对应。" % missing)
    log("样本数: 训练 %d, 验证 %d, 测试 %s" % (len(tr), len(va), {k: len(v) for k, v in tests.items()}))
    cand = np.arange(HIST, arr.shape[0] - T_OUT + 1)
    keep = np.zeros(arr.shape[0], bool)
    keep[t0_all] = True
    drop_rep = {}
    for y in have:
        a0, b0 = day_index(y) * 24, day_index(y + 1) * 24
        c = cand[(cand - HIST >= a0) & (cand + T_OUT <= b0)]
        if len(c) == 0:
            continue
        lost = c[~keep[c]]
        st = np.nan_to_num(kpmax[c], nan=0.0) >= 5
        st_lost = np.nan_to_num(kpmax[lost], nan=0.0) >= 5
        drop_rep[y] = dict(candidates=int(len(c)), dropped=int(len(lost)),
                           storm_share_all=float(st.mean()), storm_share_dropped=float(st_lost.mean()) if len(lost) else None)
        if len(lost):
            log("  %d 年: 丢弃 %d/%d 个时间窗 (%.1f%%); 磁暴窗占比 全部 %.1f%% / 被丢弃 %.1f%%" % (
                y, len(lost), len(c), 100.0 * len(lost) / len(c), 100 * st.mean(), 100 * st_lost.mean()))
    if len(tr) == 0 or len(va) == 0 or not tests:
        raise SystemExit("可用样本不足，请检查数据年份。")
    c1pg, c_have = load_maps("c1pg", TEST_Y)
    log("C1PG 基准: %s" % (c_have if c_have else "未找到，将只与持续性/3 天均值比较"))
    if a.dry:
        log("驱动量表 %s, 示例 %s" % (drv.shape, np.round(drv[tr[0]], 3)))
        return

    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    log("PyTorch %s, 设备 %s %s" % (torch.__version__, dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else ""))
    torch.backends.cudnn.benchmark = True
    bd = Bundle(torch, dev, arr, drv, doy, not a.no_solar, a.sunframe, inp=arr_in)
    if a.sunframe and a.arch != "v3":
        raise SystemExit("--sunframe 目前只支持 --arch v3")
    if a.arch == "v3":
        model = build_model(torch, use_drivers=not a.no_drivers, drop_path=a.drop_path, ndrv=drv.shape[1],
                            n_aux=bd.n_aux(), sunframe=a.sunframe).to(dev)
    elif a.arch == "convlstm":
        model = build_convlstm(torch).to(dev)
    elif a.arch == "convlstm_drv":
        model = build_convlstm(torch, use_drivers=True, ndrv=drv.shape[1]).to(dev)
    else:
        model = build_openstl_simvp(torch).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    ema = None
    if a.ema > 0:
        import copy
        ema = copy.deepcopy(model)
        for q in ema.parameters():
            q.requires_grad_(False)
    log("模型参数量 %.1f M; 配置: %s" % (nparam / 1e6, vars(a)))

    pde = None
    params = list(model.parameters())
    if a.lam_pde > 0:
        pde = torch.nn.Parameter(torch.tensor([-1.5, 0.0, 0.0, -2.0, 0.0, 0.0, -4.0], device=dev))   # q(3), beta(3), s0
        params.append(pde)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    warm = min(500, a.iters // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, it / a.iters)))))
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev.type == "cuda")
    tr_t = torch.from_numpy(tr).to(dev)
    f107n = torch.from_numpy((idx["f107"] / 150.0).astype(np.float32)).to(dev)
    eval_every = 30 if a.quick else a.eval_every
    bad = 0
    best, best_state, hist, best_it = 1e9, None, [], 0
    os.makedirs(P.CKPT, exist_ok=True)
    ck = os.path.join(P.CKPT, "ckpt_v3_%s.pt" % a.name)
    t_start = time.time()
    run_l1 = 0.0
    if a.eval_only:
        best_state = torch.load(P.ckpt(a.eval_only), map_location=dev)
        best = float("nan")
        a.iters = 0
        log("只评测模式: 载入 %s，不训练" % a.eval_only)
    for it in range(a.iters):
        t0 = tr_t[torch.randint(0, len(tr_t), (a.batch,), device=dev)]
        x, y, aux, dv = bd.batch(t0)
        with torch.autocast("cuda", enabled=a.amp and dev.type == "cuda"):
            pred = predict(torch, model, x, aux, dv, a, t0)
            d = pred - y
            l1 = d.abs().mean().detach()
            if a.loss == "l1":
                loss = d.abs().mean()
            elif a.loss == "mse":
                loss = (d * d).mean() * 25.0
            else:
                loss = d.abs().mean() + (d * d).mean() * 25.0
            if pde is not None:
                sp = torch.nn.functional.softplus
                xx = bd.x_f[None, None]
                q = sp(pde[0] + pde[1] * xx + pde[2] * xx * xx)
                b = sp(pde[3] + pde[4] * xx + pde[5] * xx * xx)
                cz = bd.cz(t0, bd.lat_f, bd.lon_f, bd.ar_out[1:-1]).clamp(min=1e-6) ** 0.7
                f = f107n[torch.clamp((t0 - 24) // 24, min=0)][:, None, None, None]
                dudt = (pred[:, 2:] - pred[:, :-2]) / 2.0
                res = dudt - (q * f * cz + sp(pde[6]) - b * pred[:, 1:-1])
                loss = loss + a.lam_pde * ((res * 6.0) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for qe, q in zip(ema.parameters(), model.parameters()):
                    qe.mul_(a.ema).add_(q.detach(), alpha=1.0 - a.ema)
        run_l1 = 0.98 * run_l1 + 0.02 * float(l1) if it else float(l1)
        if (it + 1) % eval_every == 0 or it + 1 == a.iters:
            em = ema if ema is not None else model
            r, _ = evaluate(torch, em, bd, va, kpmax, None, pde, a)
            el = (time.time() - t_start) / 60.0
            log("[%6d/%d %.1f min, 预计共 %.0f min] 训练 L1 %.3f TECU | 验证 RMSE 模型 %.3f 持续性 %.3f 3天均值 %.3f" % (
                it + 1, a.iters, el, el * a.iters / (it + 1), run_l1 * SCALE, r["model"]["rmse"], r["pers"]["rmse"], r["mean3"]["rmse"]))
            hist.append([it + 1, run_l1 * SCALE, r["model"]["rmse"]])
            if r["model"]["rmse"] < best - 1e-3:
                best, bad, best_it = r["model"]["rmse"], 0, it + 1
                best_state = {k: v.detach().clone() for k, v in em.state_dict().items()}
                torch.save(best_state, ck)
            else:
                bad += 1
                if a.patience and bad >= a.patience:
                    log("验证 RMSE 连续 %d 次评测没有改进，在第 %d 步提前停止（最好的是第 %d 步）" % (bad, it + 1, best_it))
                    break
    model.load_state_dict(best_state)
    if a.eval_only:
        ck = None
    log("最佳验证 RMSE %.3f TECU，开始测试" % best)

    out = dict(name=a.name, args=vars(a), params_M=nparam / 1e6, best_val_rmse=best, best_step=best_it, history=hist,
               train_minutes=(time.time() - t_start) / 60.0, years_available=have, c1pg_years=c_have,
               n_train=int(len(tr)), device=str(dev), torch=torch.__version__, tests={}, dropped_windows=drop_rep)
    saved_all = {}
    d0 = day_index(2024, 5, 8) * 24
    save_days = set(d0 + 24 * k for k in range(6))
    for yk, starts in tests.items():
        r, saved = evaluate(torch, model, bd, starts, kpmax, c1pg, pde, a, save_days)
        out["tests"][yk] = r
        saved_all.update(saved)
        msg = "测试 %s (%d 天): 模型 %.3f (面积加权 %.3f) | 持续性 %.3f | 3天均值 %.3f" % (
            yk, r["n_starts"], r["model"]["rmse"], r["model"]["rmse_w"], r["pers"]["rmse"], r["mean3"]["rmse"])
        if "c1pg" in r:
            msg += " | C1PG %.3f (同样本上模型 %.3f, 改进 %.1f%%)" % (
                r["c1pg"]["rmse"], r["model@c1pg"]["rmse"], 100 * (1 - r["model@c1pg"]["rmse"] / r["c1pg"]["rmse"]))
        log(msg + "  [全球 RMSE, TECU]")
    od = P.TRAIN
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "%s_metrics.json" % a.name), "w") as f:
        json.dump(out, f)
    with open(os.path.join(od, "%s_log.txt" % a.name), "w", encoding="utf-8") as f:
        f.write("\n".join(LOGS))
    if saved_all:
        np.savez_compressed(os.path.join(od, "%s_storm_maps.npz" % a.name), **saved_all)
    z = os.path.join(P.RESULTS, "results_v3_%s.zip" % a.name)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for fn in sorted(os.listdir(od)):
            if fn.startswith(a.name + "_"):
                f.write(os.path.join(od, fn), fn)
    print("完成。请把 %s 发回对话。" % z)


if __name__ == "__main__":
    main()
