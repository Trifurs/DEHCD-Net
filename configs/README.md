# Configuration Layout

The XML files use a small inheritance tree so most variables live in one place.

- `base.xml` stores shared model, optimization, logging, and inference defaults.
- `datasets/*.xml` store task definitions, dataset layout, normalization, loss, and sampling policy.
- `dehcd/*.xml` store only the DEHCD-Net size variant and run name.

Use any variant directly, for example:

```bash
python tools/train.py --config configs/dehcd/bright_l.xml
python tools/evaluate.py --config configs/dehcd/haiti_l.xml --checkpoint <checkpoint.pth>
```

Dataset roots are intentionally placeholders such as `data/BRIGHT`; edit the corresponding dataset base file or pass a modified copy for your environment.
