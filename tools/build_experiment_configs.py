"""Generate explicit experiment overlays; never launches training."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.protocol import atomic_json

DATASETS = ("bright", "haiti", "cau_flood", "xbd")
LEGACY_BASELINES = ("icif_net", "dminet", "hfa_panet", "wavehfg", "hrsicd", "haff")
DAMAGE_BASELINES = ("changeos", "damageformer", "changemamba")
BASELINES = LEGACY_BASELINES + DAMAGE_BASELINES

ABLATIONS = {
    "no_hog": {"use_hog": False},
    "hog_intensity_matched": {"use_hog": True, "hog_prior": "intensity_control"},
    # BiCSF includes BOTH GCBM and the bidirectional WASM path.
    "bicsf_conv_matched": {"gcb_mode": "conv_matched", "bicsf_mode": "conv_matched"},
    "wasm_conv_matched": {"bicsf_mode": "conv_matched"},
    "gcb_conv_matched": {"gcb_mode": "conv_matched"},
    "irb_feedforward_matched": {"irb_mode": "feedforward_matched"},
    "no_dpm": {"fusion_mode": "plain"},
    "dpm_conv_matched": {"fusion_mode": "plain_matched"},
    "no_flow": {"align_fusion": False},
    "no_difference_gate": {"difference_gate": False},
    "no_flow_no_gate": {"align_fusion": False, "difference_gate": False},
    "no_modality_gate": {"adaptive_modality_weight": False},
    "no_bicsf": {"global_context": False, "cross_scale_fusion": False},
    "no_wasm": {"cross_scale_fusion": False},
    "no_gcb": {"global_context": False},
    "no_irb": {"diffusion_steps": 0},
    "all_off": {"use_hog": False, "fusion_mode": "plain", "cross_scale_fusion": False,
                "global_context": False, "diffusion_steps": 0},
}


def xml_write(path, config):
    def add(parent, mapping):
        for key, value in mapping.items():
            elem = ET.SubElement(parent, key)
            if isinstance(value, dict):
                add(elem, value)
            else:
                typename = {bool: "bool", int: "int", float: "float", list: "list"}.get(type(value), "str")
                elem.set("type", typename)
                elem.text = str(value).lower() if isinstance(value, bool) else repr(value) if isinstance(value, list) else str(value)
    root = ET.Element("project")
    add(root, config)
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate(root=ROOT):
    catalog = []
    def emit(dataset, suffix, suite, overrides, *, reference=None, factor=None,
             kind="control", allowed_changes=None):
        identifier = f"{dataset}_{suffix}"
        if reference is None and suffix != "dehcd_l":
            reference = f"{dataset}_dehcd_l"
        if allowed_changes is None:
            allowed_changes = [f"{section}.{key}" for section in ("model", "training", "dataset")
                               for key in overrides.get(section, {})]
        config = {"base_param": f"../../dehcd/{dataset}_l.xml",
                  "model": {"dropout": .15, "pretrained_backbone": False,
                            "encoder_checkpoint": "", "deep_supervision": False},
                  "training": {"early_stop_patience": 0, "deterministic": True, "amp": False,
                               "localization_loss_weight": 0.0, "aux_loss_weight": 0.0,
                               "feature_pair_loss_weight": 0.0, "best_metric": "foreground_miou"},
                  "inference": {"test_time_augmentation": "none", "prediction_rule": "damage_argmax"},
                  "experiment": {"id": identifier, "dataset": dataset, "suite": suite,
                                 "protocol": "controlled_comparison_v1", "status": "configured",
                                 "comparison": {"kind": kind, "reference": reference or "",
                                                "factor": factor or suffix, "allowed_changes": allowed_changes}},
                  "logging": {"run_name": identifier}}
        for key, value in overrides.items():
            config.setdefault(key, {}).update(value)
        path = root / "configs" / "experiments" / suite / f"{identifier}.xml"
        xml_write(path, config)
        catalog.append({"id": identifier, "dataset": dataset, "suite": suite,
                        "config": str(path.relative_to(root))})

    def baseline(ds, name, bn=True):
        return {"name": name, "compare_model": name, "compare_adapt_batchnorm": bn,
                "compare_deep_supervision": False, "compare_target_channels": 3,
                "encoder_checkpoint": "", "selective_scan_backend": "auto",
                "changemamba_variant": "bda" if ds == "xbd" else "multimodal",
                "share_damage_encoder": ds == "xbd"}

    for ds in DATASETS:
        for size, channels in (("s", 16), ("m", 24), ("l", 32)):
            emit(ds, f"dehcd_{size}", "main", {"model": {"backbone": f"dehcd_{size}", "base_channels": channels}},
                 kind="architecture", factor="architecture", allowed_changes=["model"])
        for name in BASELINES:
            emit(ds, name, "main", {"model": baseline(ds, name),
                 "experiment": {"implementation": "adapted_official_core", "adapter": "learned_input_GN_shared_primary_objective"}},
                 kind="architecture", factor="architecture", allowed_changes=["model"])
            emit(ds, f"{name}_original_bn", "adapter", {"model": baseline(ds, name, bn=False),
                 "experiment": {"implementation": "adapted_input_and_head_original_BN"}},
                 reference=f"{ds}_{name}", factor="batchnorm_adaptation",
                 allowed_changes=["model.compare_adapt_batchnorm"])
        for name in DAMAGE_BASELINES:
            emit(ds, f"{name}_localization_aux", "auxiliary", {"model": baseline(ds, name),
                 "training": {"localization_loss_weight": 1.0}},
                 reference=f"{ds}_{name}", factor="localization_auxiliary_supervision",
                 allowed_changes=["training.localization_loss_weight"])
    # Head objective changes ONLY, relative to the same model's original-BN
    # control. Sampling, preprocessing, selection and budget remain fixed.
    for ds in ("bright", "xbd"):
        for name in DAMAGE_BASELINES:
            emit(ds, f"{name}_head_recipe", "head_recipe", {
                "model": baseline(ds, name, bn=False),
                "training": {"loss": "changeos_native" if name == "changeos" else "damage_ce_lovasz",
                             "localization_loss_weight": 1.0},
                "experiment": {"implementation": "official_architecture_project_input_native_head_loss",
                               "pretraining": "scratch", "protocol_scope": "head_recipe_control_not_official_benchmark"}},
                reference=f"{ds}_{name}_original_bn", factor="head_objective",
                allowed_changes=["training.loss", "training.localization_loss_weight"])
    for ds in ("bright", "xbd"):
        for name in ("damageformer", "changemamba"):
            emit(ds, f"{name}_optimizer_recipe", "optimizer_recipe", {
                "model": baseline(ds, name, bn=False),
                "training": {"loss": "damage_ce_lovasz", "optimizer": "adamw", "learning_rate": 1e-4,
                             "weight_decay": 5e-3, "scheduler": "none", "localization_loss_weight": 1.0},
                "experiment": {"implementation": "official_head_and_optimizer_recipe_project_data_and_budget",
                               "pretraining": "scratch", "protocol_scope": "reference_optimizer_control_not_official_benchmark"}},
                reference=f"{ds}_{name}_head_recipe", factor="optimizer_recipe", kind="recipe_bundle",
                allowed_changes=["training.optimizer", "training.learning_rate", "training.weight_decay", "training.scheduler"])
    for ds in ("bright", "haiti"):
        for name, switches in ABLATIONS.items():
            emit(ds, name, "ablation", {"model": switches})
        for bins in (2, 4, 6, 8, 10):
            emit(ds, f"hog_bins_{bins}", "sensitivity", {"model": {"hog_bins": bins}})
        for steps in (0, 1, 2, 3, 4):
            emit(ds, f"irb_steps_{steps}", "sensitivity", {"model": {"diffusion_steps": steps}})
        simple_loss = {"loss": "compound", "ce_weight": 1.0, "dice_weight": 1.0,
                       "focal_weight": 0.0, "lovasz_weight": 0.0, "tversky_weight": 0.0,
                       "foreground_dice_weight": 0.0, "label_smoothing": 0.0,
                       "aux_loss_weight": 0.0, "feature_pair_loss_weight": 0.0}
        emit(ds, "basic_ce_dice", "recipe", {"training": {**simple_loss, "class_weights": [], "class_balanced_sampler": False},
             "dataset": {"positive_crop_prob": 0.0, "rare_crop_prob": 0.0, "crop_candidate_count": 1}}, kind="recipe_bundle", factor="basic_training_recipe")
        emit(ds, "full_recipe", "recipe", {})
        emit(ds, "ce_dice_full_sampling", "recipe", {"training": simple_loss})
        emit(ds, "no_weighted_sampler", "recipe", {"training": {"class_balanced_sampler": False}})
        emit(ds, "no_targeted_crop", "recipe", {"dataset": {"positive_crop_prob": 0.0, "rare_crop_prob": 0.0, "crop_candidate_count": 1}})
        weights = {"class_weights": []}
        if ds == "haiti":
            weights.update(hier_binary_class_weights=[1., 1.], hier_subclass_weights=[1., 1., 1.])
        emit(ds, "no_class_weights", "recipe", {"training": weights})
        if ds == "haiti":
            # Keep historical and one-factor controls for the explicit change
            # in foreground/background reduction; never mix these run results.
            emit(ds, "legacy_pixel_mean", "recipe", {"training": {
                "hier_binary_reduction": "pixel_mean", "hier_binary_class_weights": [1., 2.8]}},
                reference=f"{ds}_pixel_mean_equal_weights", factor="binary_class_weights",
                allowed_changes=["training.hier_binary_class_weights"])
            emit(ds, "pixel_mean_equal_weights", "recipe", {"training": {
                "hier_binary_reduction": "pixel_mean", "hier_binary_class_weights": [1., 1.]}})
    atomic_json(root / "configs/experiments/catalog.json", {"schema_version": 2, "experiments": catalog})
    return catalog


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(f"Generated {len(generate())} experiment overlays under configs/experiments/")
