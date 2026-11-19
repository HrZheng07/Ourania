# Ourania

全球电离层 TEC 图 24 小时预报模型。

- 输入：过去 48 小时的 CODE 全球 TEC 图（逐小时，2.5°×5°）和 17 个太阳/地磁驱动量
- 输出：未来 24 小时的全球 TEC 图
- 参数量：1030 万

## 权重

| 文件 | 说明 |
|---|---|
| `checkpoints/ckpt_v3_rapin*.pt` | 快速版：输入用 CODE 快速图训练（3 个种子） |
| `checkpoints/ckpt_v3_ema_dp10*.pt` | 主模型：输入用 CODE 最终图训练（3 个种子） |

快速版的输入和 CODE 1 天预报（C1PG）发布时能拿到的数据相同。2026 年 1–11 月全球 RMSE（真值为 CODE 最终图）：快速版 3.55 TECU，C1PG 4.29 TECU。

## 使用

```bash
pip install -r requirements.txt    # torch 按 CUDA 版本另装

# 下载数据（放到 data/）
python code/download_indices.py
python code/download_gim_global.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
python code/download_rapid_cddis.py 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
python code/download_c1pg_cddis.py 2024 2025

# 评测
python code/evaluate.py --ckpt ckpt_v3_rapin.pt --input rap

# 训练
python code/ourania.py --name rapin --input-product rap --amp
```

CDDIS 的下载需要 NASA Earthdata 账号，运行时输入，不保存。
