# Ourania

全球电离层 TEC 图 24 小时预报模型。

- 输入：过去 48 小时的 CODE 全球 TEC 图（逐小时，2.5°×5°）和 17 个太阳/地磁驱动量
- 输出：未来 24 小时的全球 TEC 图
- 参数量：1030 万

## 权重（文件体积过大，需要者请邮箱联系我）



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
