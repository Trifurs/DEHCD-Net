"""Dual-head diagnostic metrics, kept separate from full-map IoU comparisons."""
from utils.metrics import ConfusionMatrixMeter
from utils.losses import localization_target


class DamageHeadMeter:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.localization = ConfusionMatrixMeter(2, ignore_index)
        self.damage = ConfusionMatrixMeter(num_classes, ignore_index)
        self.seen = False

    def update(self, output, target):
        if not isinstance(output, dict) or 'localization_logits' not in output:
            return
        self.seen = True
        self.localization.update(output['localization_logits'].argmax(1), localization_target(target, self.ignore_index))
        valid_foreground = (target > 0) & (target != self.ignore_index)
        self.damage.update(output['logits'].argmax(1)[valid_foreground], target[valid_foreground])

    def compute(self):
        if not self.seen:
            return None
        metrics = self.damage.compute()
        f1 = [metrics[f'class_{i}_f1'] for i in range(1, self.num_classes)]
        return {'localization_f1': self.localization.compute()['f1'],
                'conditional_damage_hmean_f1': (len(f1) / sum(1 / max(x, 1e-7) for x in f1)
                                               if metrics['valid_pixels'] > 0 else None),
                'conditional_damage_f1_per_class': f1,
                'labeled_foreground_pixels': metrics['valid_pixels'],
                'localization_confusion_matrix': self.localization.matrix.long().tolist(),
                'damage_on_labeled_foreground_confusion_matrix': self.damage.matrix.long().tolist(),
                'scope': 'Damage classification restricted to ground-truth foreground; harmonic F1 clips each class at 1e-7 as upstream. Not full-map mIoU or an independent generalization estimate.'}
