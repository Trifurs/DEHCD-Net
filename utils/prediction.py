"""Identical TTA and explicitly declared decoding for train/eval/test/inference."""
from contextlib import nullcontext


def predict_outputs(model, optical, sar, amp: bool, tta_mode: str):
    import torch
    from utils.model_outputs import extract_logits

    def forward_once(optical_batch, sar_batch):
        context = torch.amp.autocast('cuda', enabled=True) if amp else nullcontext()
        with context:
            out = model(optical_batch, sar_batch)
        heads = {'logits': extract_logits(out)}
        if isinstance(out, dict) and 'localization_logits' in out:
            heads['localization_logits'] = out['localization_logits']
        return heads

    if tta_mode == 'none':
        return forward_once(optical, sar)
    transforms = []
    if tta_mode == 'flips':
        for fh, fv in [(False, False), (True, False), (False, True), (True, True)]:
            fn = lambda x, fh=fh, fv=fv: _flip_tensor(x, fh, fv)
            transforms.append((fn, fn))
    elif tta_mode == 'd4':
        for k in range(4):
            for fh in (False, True):
                transforms.append((lambda x, k=k, fh=fh: _apply_d4(x, k, fh),
                                   lambda x, k=k, fh=fh: _invert_d4(x, k, fh)))
    else:
        raise ValueError(f'Unsupported test-time augmentation: {tta_mode}')
    outputs = [ {key: inverse(value).float() for key, value in forward_once(forward(optical), forward(sar)).items()}
                for forward, inverse in transforms ]
    return {key: torch.stack([out[key] for out in outputs]).mean(0) for key in outputs[0]}


def predict_logits(model, optical, sar, amp: bool, tta_mode: str):
    return predict_outputs(model, optical, sar, amp, tta_mode)['logits']


def _flip_tensor(tensor, flip_h: bool, flip_v: bool):
    dims = ([-2] if flip_v else []) + ([-1] if flip_h else [])
    return tensor.flip(dims=dims) if dims else tensor


def _apply_d4(tensor, rot_k: int, flip_h: bool):
    out = tensor.rot90(int(rot_k), dims=(-2, -1))
    return out.flip(dims=(-1,)) if flip_h else out


def _invert_d4(tensor, rot_k: int, flip_h: bool):
    out = tensor.flip(dims=(-1,)) if flip_h else tensor
    return out.rot90(-int(rot_k), dims=(-2, -1))


def resolve_tta(config, override=None):
    cfg = config.get('inference', {})
    mode = str(override if override is not None else cfg.get('test_time_augmentation', cfg.get('tta', 'none'))).lower()
    if mode in {'', 'off', 'false', 'none'}:
        mode = 'none'
    if mode not in {'none', 'flips', 'd4'}:
        raise ValueError(f'Unsupported test-time augmentation: {mode}')
    return mode


def prediction_rule(config):
    rule = str(config.get('inference', {}).get('prediction_rule', 'damage_argmax'))
    if rule not in {'damage_argmax', 'localization_gated', 'changeos_object'}:
        raise ValueError(f'Unsupported prediction rule: {rule}')
    return rule


def decode_predictions(output, config):
    import torch
    from utils.model_outputs import extract_logits
    logits = extract_logits(output)
    pred = logits.argmax(1)
    rule = prediction_rule(config)
    if rule == 'damage_argmax':
        return pred
    if not isinstance(output, dict) or 'localization_logits' not in output:
        raise ValueError(f'{rule} requires localization_logits')
    loc = output['localization_logits'].argmax(1).bool()
    if rule == 'localization_gated':
        return torch.where(loc, pred, 0)
    import numpy as np
    import cv2
    weights = config.get('inference', {}).get('object_class_weights')
    if weights is None and logits.shape[1] == 5:
        weights = [8., 38., 25., 11.]  # Original xBD classes 1..4 only.
    if weights is None or len(weights) != logits.shape[1] - 1 or any(float(w) <= 0 for w in weights):
        raise ValueError('changeos_object needs positive weights for every foreground class; '
                         'the original xBD weights cannot be reused for BRIGHT class definitions')
    result = []
    for mask, damage in zip(loc.cpu().numpy(), pred.cpu().numpy()):
        count, regions = cv2.connectedComponents(mask.astype('uint8'), connectivity=8)
        refined = np.zeros_like(damage)
        # Vote only real components (not background), fixing the upstream loop's
        # region-id/boolean-mask comparison for the empty-localization case.
        for region in range(1, count):
            pixels = regions == region
            votes = np.bincount(damage[pixels], minlength=logits.shape[1])[1:] * np.asarray(weights)
            refined[pixels] = int(votes.argmax()) + 1
        result.append(torch.from_numpy(refined))
    return torch.stack(result).to(logits.device)
