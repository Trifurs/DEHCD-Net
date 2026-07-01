# Config Profiles

The XML configurations are split by hardware profile:

- `2080ti_x2/`: original two-GPU RTX 2080Ti settings.
- `5090_x1/`: single RTX 5090 32GB settings.

Use explicit paths when running experiments, for example:

```bash
python tools/train.py --config configs/5090_x1/bright_multiclass_dehcd_s.xml
python tools/train.py --config configs/2080ti_x2/bright_multiclass_dehcd_s.xml
```

`5090_x1` configs use `cuda:0`, `multi_gpu=single`, `gpu_ids=[0]`, and
larger per-GPU batches chosen separately for BRIGHT1, CAU1, DEHCD-Net variants,
and the official comparison models.

5090 x1 batch policy:

| Dataset/Profile | DEHCD-S | DEHCD-M | DEHCD-L | Notes |
| --- | ---: | ---: | ---: | --- |
| BRIGHT1 256 | 24 | 18 | 12 | Heavy official models vary by memory footprint: 6 to 24. |
| CAU1 256 | 24 | 18 | 12 | Heavy official models use batch 6 to 24. |
| Haiti1 128 | 48 | 36 | 24 | Uses pre-event S2 bands 0/1/2 and post-event S1 bands 0/1 from both passes; S2 band 3 and S1 band 2 are quality masks used for normalization/ignore policy, not model input channels. |

The 2080Ti x2 profile keeps the same run names while using the updated
1.5x batch-size policy where memory allows.
