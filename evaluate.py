# -*- coding: utf-8 -*-
"""载入检查点做评测: 逐日误差、磁暴分组、与 C1PG 比较、bootstrap 置信区间。

用法:
    python code/evaluate.py --ckpt ckpt_v3_ema_dp10.pt --input fin rap
    python code/evaluate.py --ckpt ckpt_v3_rapin.pt --input rap
    python code/evaluate.py --ckpt ckpt_v3_rapin.pt --input rap --final    # 2025 年
"""
import argparse
import json
import math
import os
from datetime import datetime

import numpy as np

import ourania as V
import paths as P

HERE = V.HERE
CN_LAT = (V.LAT[:71] >= 15) & (V.LAT[:71] <= 55)
CN_LON = (V.LON >= 70) & (V.LON <= 140)
WL = V.W_LAT[:, None]                       # (71,1) 面积权重, 均值 1


def year_window(prefix, y):
    """把 prefix_y.npz 放进该年的小时轴 (nd*24, 72, 72)，缺失 -1；文件不存在返回 None。"""
    p = P.data("%s_%d.npz" % (prefix, y))
    if not os.path.exists(p):
        return None
    nd = V.day_index(y + 1) - V.day_index(y)
    w = np.full((nd * 24, V.H, V.W), -1, dtype=np.int16)
    t = np.load(p)["tec"][:nd * 24]
    w[:t.shape[0], :71] = t[:, :, :72]
    w[:t.shape[0], 71] = t[:, 70, :72]
    return w


def issue_lag_hours(y):
    """C1PG 每天文件的生成时间相对目标日 0 UT 的小时数 (按 doy)。"""
    p = P.data("c1pg_%d_issue.json" % y)
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))["issued"]
    out = {}
    for doy, s in d.items():
        if not s:
            continue
        t = datetime.strptime(s, "%d-%b-%y %H:%M")
        out[int(doy)] = (t - datetime(y, 1, 1)).total_seconds() / 3600.0 - 24.0 * (int(doy) - 1)
    return out


def block_boot(fn, n, B=2000, L=7, seed=0):
    """按天分块 bootstrap, 返回 2.5%/97.5% 分位数和 fn<=0 的比例。"""
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / L))
    vals = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n - L + 1, nb)
        idx = (st[:, None] + np.arange(L)[None]).ravel()[:n]
        vals[b] = fn(idx)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float((vals <= 0).mean())


def day_stats(e):
    """每天每个时效的误差统计。"""
    e2 = e * e
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        return dict(mse=np.nanmean(e2, axis=(2, 3)), msew=np.nanmean(e2 * WL, axis=(2, 3)),
                    msecn=np.nanmean(e2[:, :, CN_LAT][:, :, :, CN_LON], axis=(2, 3)), bias=np.nanmean(e, axis=(2, 3)))


def make_predictor(a, arr_w, drv_w, doy_w):
    """返回 f(起点) -> 预报 [B,24,71,72]; --selftest 时用简单预报代替模型。"""
    if a.selftest:
        def f(t):
            pers = np.stack([arr_w[s - 24:s] for s in t]).astype(np.float32)[:, :, :71] / 10.0
            m3 = np.stack([(arr_w[s - 24:s].astype(np.float32) + arr_w[s - 48:s - 24] + arr_w[s - 72:s - 48]) / 3.0
                           for s in t])[:, :, :71] / 10.0
            return 0.5 * (pers + m3)
        return f
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bd = V.Bundle(torch, dev, arr_w, drv_w, doy_w, not a.no_solar, a.sunframe)
    if a.arch == "v3":
        model = V.build_model(torch, use_drivers=not a.no_drivers, ndrv=drv_w.shape[1], n_aux=bd.n_aux(), sunframe=a.sunframe)
    elif a.arch == "convlstm":
        model = V.build_convlstm(torch)
    elif a.arch == "convlstm_drv":
        model = V.build_convlstm(torch, use_drivers=True, ndrv=drv_w.shape[1])
    else:
        model = V.build_openstl_simvp(torch)
    model = model.to(dev)
    model.load_state_dict(torch.load(P.ckpt(a.ckpt), map_location=dev))
    model.eval()

    def f(t):
        with torch.no_grad():
            tt = torch.as_tensor(t, device=dev)
            x, _, aux, dv = bd.batch(tt)
            p = V.predict(torch, model, x, aux, dv, a, tt)
            return (p[:, :, :71] * V.SCALE).float().cpu().numpy()
    f.cleanup = lambda: (bd.__dict__.clear(), torch.cuda.empty_cache() if dev.type == "cuda" else None)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="", help="输出文件名里的名字, 默认取检查点名去掉 ckpt_v3_")
    ap.add_argument("--arch", default="v3", choices=["v3", "convlstm", "convlstm_drv", "openstl_simvp"])
    ap.add_argument("--no-drivers", action="store_true")
    ap.add_argument("--no-solar", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--sunframe", action="store_true")
    ap.add_argument("--sw", action="store_true")
    ap.add_argument("--input", nargs="+", default=["fin"], choices=["fin", "rap", "rt", "rtc"],
                    help="fin = CODE 最终图; rap = CODE 快速图; rt = IGS 联合实时图 IRTG; rtc = 校准后的实时图 (先运行 download_irtg.py 和 build_rt_inputs.py)")
    ap.add_argument("--issue-hour", type=int, default=0, choices=range(24), help="起报时刻 (UT 小时), 默认 0")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--truth", default="final", choices=["final", "igs", "jpl", "esa"],
                    help="真值产品: final = CODE 最终图 (默认); igs/jpl/esa = 其他机构最终图 (alt_<产品>_<年>.npz, 2 小时一张, 只在有图的时刻计算)")
    ap.add_argument("--final", action="store_true", help="评测保留的 2025 年（模型定型后只跑一次）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--lag-max", type=float, default=12.0, help="C1PG 发布时间敏感性: 只保留生成时间不晚于目标日 0 UT 后这么多小时的天")
    ap.add_argument("--selftest", action="store_true", help="不载入模型, 用简单数值预报代替, 只检查数据流程")
    a = ap.parse_args()
    if a.year == 2025 and not a.final:
        raise SystemExit("2025 年是保留的最终测试年, 模型定型后才加 --final 评测。")
    name = a.name or os.path.splitext(os.path.basename(a.ckpt))[0].replace("ckpt_v3_", "")
    if a.selftest:
        name = "selftest"
    tag = "" if a.truth == "final" else "_" + a.truth
    if a.issue_hour:
        tag += "_h%02d" % a.issue_hour
    if any(k in ("rt", "rtc") for k in a.input):
        tag += "_rt"          # 实时图实验单独存文件, 不覆盖主结果
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    y = a.year
    o = V.day_index(y) * 24
    nd = V.day_index(y + 1) - V.day_index(y)
    L = nd * 24

    # ---- 数据: 只放测试年这一段, 省内存 ----
    fin = year_window("gim", y)
    if fin is None:
        raise SystemExit("缺少 gim_%d.npz" % y)
    inputs = {}
    for k in a.input:
        w = fin if k == "fin" else year_window(k, y)
        if w is None:
            raise SystemExit("缺少 %s_%d.npz, 先运行 python %s %d" % (k, y, "download_rapid_cddis.py" if k == "rap" else "download_irtg.py 和 build_rt_inputs.py", y))
        inputs[k] = w
    c1pg = year_window("c1pg", y)
    if a.truth == "final":
        tru = fin
    else:
        tru = year_window("alt_" + a.truth, y)
        if tru is None:
            raise SystemExit("缺少 alt_%s_%d.npz, 先运行 download_alt_gim.py" % (a.truth, y))
    lag = issue_lag_hours(y)
    n_all = V.day_index(y + 2) * 24
    idx = V.load_indices(n_all // 24 + 1)
    drv, doy, kp_during = V.driver_table(idx, n_all)
    if a.sw:
        drv = np.concatenate([drv, V.sw_table(V.load_omni(n_all), n_all)], axis=1)
    k0 = np.arange(n_all) // 3
    kp_before = np.array([np.nanmax(idx["kp"][k - 8:k]) if k >= 8 and np.isfinite(idx["kp"][k - 8:k]).any() else np.nan for k in k0])
    drv_w, doy_w = drv[o:o + L], doy[o:o + L]

    # ---- 起报日: 与 ourania.py 相同, 再要求每种输入的 72 小时历史都完整 ----
    ok_f = fin.reshape(L, -1).min(axis=1) >= 0
    cand = np.arange(V.HIST + a.issue_hour, L - V.T_OUT + 1, 24)
    starts = [s for s in cand if ok_f[s - V.HIST:s + V.T_OUT].all()]
    n_fin_only = len(starts)
    for k, w in inputs.items():
        okw = w.reshape(L, -1).min(axis=1) >= 0
        starts = [s for s in starts if okw[s - V.HIST:s].all()]
    starts = np.array(starts, dtype=np.int64)
    print("%d 年: 起报日 %d 个 (只看 FIN 时 %d 个); 输入: %s" % (y, len(starts), n_fin_only, " / ".join(a.input)))
    if len(starts) < 30:
        raise SystemExit("可用起报日太少, 请检查数据。")

    # ---- 逐日误差 ----
    out = dict(starts=starts + o, kp_during=kp_during[starts + o], kp_before=kp_before[starts + o],
               c1pg_lag_h=np.array([lag.get(int(s // 24) + 1, np.nan) for s in starts]))
    def truth(t):
        v = np.stack([tru[s:s + 24] for s in t]).astype(np.float32)[:, :, :71]
        v[v < 0] = np.nan
        return v / 10.0
    if c1pg is not None:
        rows = {k: [] for k in ("mse", "msew", "msecn", "bias")}
        has = []
        for i in range(0, len(starts), 64):
            t = starts[i:i + 64]
            cc = np.stack([c1pg[s:s + 24] for s in t])
            h = cc.reshape(len(t), -1).min(axis=1) >= 0
            st = day_stats(cc.astype(np.float32)[:, :, :71] / 10.0 - truth(t))
            for k in rows:
                v = st[k]
                v[~h] = np.nan
                rows[k].append(v)
            has.append(h)
        for k in rows:
            out["%s_c1pg" % k] = np.concatenate(rows[k])
        out["c1pg_ok"] = np.concatenate(has)
    for inp, w in inputs.items():
        f = make_predictor(a, w, drv_w, doy_w)
        acc = {(m, k): [] for m in ("model", "pers", "mean3") for k in ("mse", "msew", "msecn", "bias")}
        for i in range(0, len(starts), a.batch):
            t = starts[i:i + a.batch]
            yt = truth(t)
            pers = np.stack([w[s - 24:s] for s in t]).astype(np.float32)[:, :, :71] / 10.0
            m3 = np.stack([(w[s - 24:s].astype(np.float32) + w[s - 48:s - 24] + w[s - 72:s - 48]) / 3.0 for s in t])[:, :, :71] / 10.0
            for m, pr in (("model", f(t)), ("pers", pers), ("mean3", m3)):
                st = day_stats(pr - yt)
                for k in ("mse", "msew", "msecn", "bias"):
                    acc[(m, k)].append(st[k])
        for (m, k), v in acc.items():
            out["%s_%s_%s" % (k, m, inp)] = np.concatenate(v)
        if hasattr(f, "cleanup"):
            f.cleanup()
        print("  输入 %s 完成" % inp)
    od = P.EVAL
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, "daily_%s_%d%s.npz" % (name, y, tag)), **out)

    # ---- 汇总 ----
    rm = lambda v, sel: math.sqrt(float(np.nanmean(v[sel]))) if sel.any() else float("nan")
    dmean = lambda arr: np.nanmean(arr, axis=1)
    ok = out.get("c1pg_ok", np.zeros(len(starts), bool))
    groups = dict(all=np.ones(len(starts), bool),
                  storm_during=np.nan_to_num(out["kp_during"]) >= 5, quiet_during=np.nan_to_num(out["kp_during"]) < 5,
                  storm_before=np.nan_to_num(out["kp_before"]) >= 5,
                  lag_ok=np.nan_to_num(out["c1pg_lag_h"], nan=1e9) <= a.lag_max)
    summ = dict(ckpt=a.ckpt, year=y, truth=a.truth, n_days=int(len(starts)), n_c1pg_days=int(ok.sum()), inputs=a.input,
                groups={g: int(s.sum()) for g, s in groups.items()}, results={})
    print("\n===== %d 年, 真值 = %s, RMSE (TECU); 括号里是起报日数 =====" % (y, "CODE 最终图" if a.truth == "final" else a.truth.upper() + " 最终图"))
    print("%-14s %-5s %8s %8s %8s %8s %9s" % ("分组", "输入", "模型", "持续性", "3天均值", "C1PG", "相对C1PG"))
    for inp in a.input:
        R = {}
        for g, sel in groups.items():
            s = sel & ok if ok.any() else sel
            r = dict(n=int(s.sum()))
            for m in ("model", "pers", "mean3"):
                v = dmean(out["mse_%s_%s" % (m, inp)])
                r[m] = rm(v, s)
            r["model_w"] = rm(dmean(out["msew_model_%s" % inp]), s)
            r["model_china"] = rm(dmean(out["msecn_model_%s" % inp]), s)
            if ok.any():
                r["c1pg"] = rm(dmean(out["mse_c1pg"]), s)
                r["c1pg_w"] = rm(dmean(out["msew_c1pg"]), s)
                r["c1pg_china"] = rm(dmean(out["msecn_c1pg"]), s)
                r["improve_vs_c1pg"] = 1 - r["model"] / r["c1pg"]
            R[g] = r
            print("%-14s %-5s %8.3f %8.3f %8.3f %8s %8s  (%d)" % (
                g, inp, r["model"], r["pers"], r["mean3"], "%.3f" % r["c1pg"] if "c1pg" in r else "-",
                "%.1f%%" % (100 * r["improve_vs_c1pg"]) if "improve_vs_c1pg" in r else "-", r["n"]))
        R["rmse_by_lead_model"] = np.sqrt(np.nanmean(out["mse_model_%s" % inp][ok if ok.any() else slice(None)], axis=0)).tolist()
        if ok.any():
            ii = np.where(ok)[0]
            dm = dmean(out["mse_model_%s" % inp])[ii]
            dc = dmean(out["mse_c1pg"])[ii]
            lo, hi, p = block_boot(lambda j: 1 - math.sqrt(np.nanmean(dm[j])) / math.sqrt(np.nanmean(dc[j])), len(ii), a.boot)
            R["improve_vs_c1pg_ci95"] = [lo, hi]
            R["improve_vs_c1pg_p_le0"] = p
            print("   %s 输入: 相对 C1PG 改进率的 95%% 置信区间 [%.1f%%, %.1f%%] (分块 bootstrap, 块长 7 天)" % (inp, 100 * lo, 100 * hi))
        summ["results"][inp] = R
    for k2 in [k for k in a.input if k != "fin"] if "fin" in inputs else []:
        df = dmean(out["mse_model_fin"])
        dr = dmean(out["mse_model_%s" % k2])
        lo, hi, _ = block_boot(lambda j: math.sqrt(np.nanmean(dr[j])) - math.sqrt(np.nanmean(df[j])), len(df), a.boot)
        d = math.sqrt(np.nanmean(dr)) - math.sqrt(np.nanmean(df))
        summ["%s_minus_fin_rmse" % k2] = dict(value=d, ci95=[lo, hi])
        print("%s 输入相对最终图输入: RMSE 变化 %+.3f TECU, 95%% 置信区间 [%+.3f, %+.3f]" % ({"rap": "快速图", "rt": "实时图", "rtc": "校准实时图"}[k2], d, lo, hi))
        lead_f = np.sqrt(np.nanmean(out["mse_model_fin"], axis=0))
        lead_r = np.sqrt(np.nanmean(out["mse_model_%s" % k2], axis=0))
        hh = (0, 5, 11, 23) if a.truth == "final" else (0, 6, 12, 22)
        print("逐时效 (第 %s 小时) FIN 输入 %s | %s 输入 %s" % ("/".join(str(h + 1) for h in hh),
            " ".join("%.2f" % lead_f[h] for h in hh), k2, " ".join("%.2f" % lead_r[h] for h in hh)))
    fn = os.path.join(od, "ext_%s_%d%s.json" % (name, y, tag))
    with open(fn, "w", encoding="utf-8") as fo:
        json.dump(summ, fo, indent=1, ensure_ascii=False)
    # 与训练时的测试结果核对 (FIN 输入应当完全一致)
    mj = os.path.join(P.TRAIN, "%s_metrics.json" % name)
    tin = json.load(open(mj)).get("args", {}).get("input_product", "fin") if os.path.exists(mj) else "fin"
    if tin in a.input and os.path.exists(mj) and not a.selftest and a.truth == "final" and a.issue_hour == 0:
        ref = json.load(open(mj)).get("tests", {}).get(str(y), {}).get("model", {}).get("rmse")
        allf = math.sqrt(float(out["mse_model_%s" % tin].mean()))
        if ref and len(starts) == n_fin_only:
            print("核对: 本脚本 %s 输入的模型 RMSE %.4f, 训练时记录 %.4f %s" % (tin.upper(), allf, ref, "✓" if abs(allf - ref) < 2e-3 else "✗ 不一致, 请告诉我"))
    print("已保存:", fn, "和 daily_%s_%d%s.npz" % (name, y, tag))


if __name__ == "__main__":
    main()
