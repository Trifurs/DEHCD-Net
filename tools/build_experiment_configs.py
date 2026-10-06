"""Generate canonical experiment definitions and their derived analysis views."""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.experiment_catalog import derive_catalog, write_catalog_views

DATASETS = ("bright", "haiti", "xbd", "cau_flood")
LEGACY_BASELINES = ("icif_net", "dminet", "hfa_panet", "wavehfg", "hrsicd", "haff")
DAMAGE_BASELINES = ("changeos", "damageformer", "changemamba")
SEEDS = (42, 1051, 2060)
TABLES = {"bright": "Table 2", "haiti": "Table 3", "cau_flood": "Table 4", "xbd": "Table 5"}
COMPONENTS = {
    "no_dpm": {"fusion_mode": "plain"},
    "no_bicsf": {"global_context": False, "cross_scale_fusion": False},
    "no_irb": {"diffusion_steps": 0},
    "no_hog": {"use_hog": False},
    "no_hog_no_dpm": {"use_hog": False, "fusion_mode": "plain"},
    "no_hog_no_dpm_no_bicsf": {"use_hog": False, "fusion_mode": "plain",
                               "global_context": False, "cross_scale_fusion": False},
    "all_off": {"use_hog": False, "fusion_mode": "plain", "global_context": False,
                "cross_scale_fusion": False, "diffusion_steps": 0},
}
CAPACITY = {
    "hog_intensity_matched": {"use_hog": True, "hog_prior": "intensity_control"},
    "dpm_conv_matched": {"fusion_mode": "plain_matched"},
    "bicsf_conv_matched": {"gcb_mode": "conv_matched", "bicsf_mode": "conv_matched"},
    "irb_feedforward_matched": {"irb_mode": "feedforward_matched"},
}
DPM = {"no_flow": {"align_fusion": False}, "no_difference_gate": {"difference_gate": False}}


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
    tree = ET.Element("project")
    add(tree, config)
    ET.indent(tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(tree).write(path, encoding="utf-8", xml_declaration=True)


def use(group, role, paper_refs=(), **fields):
    return {"group": group, "role": role, "paper_refs": list(paper_refs), **fields}


def generate(root=ROOT):
    root = Path(root)
    paths = []

    def emit(ds, suffix, suite, overrides, *, variant="l", uses=(), roles=(),
             reference=None, factor=None, kind="control", allowed_changes=None):
        identifier = f"{ds}_{suffix}"
        if reference is None and suffix != "dehcd_l":
            reference = f"{ds}_dehcd_l"
        if allowed_changes is None:
            allowed_changes = [f"{section}.{key}" for section in ("model", "training", "dataset")
                               for key in overrides.get(section, {})]
        # Controls inherit their unique main definition. M scans cannot silently
        # inherit L architecture settings through a generic full-model template.
        config = {"base_param": f"../../dehcd/{ds}_l.xml" if suite == "main"
                  else f"../main/{reference}.xml"}
        if suite == "main":
            config.update({
                "model": {"dropout": .15, "pretrained_backbone": False,
                          "encoder_checkpoint": "", "deep_supervision": False},
                "training": {"early_stop_patience": 100 if ds == "haiti" else 30,
                             "early_stop_min_delta": 0.001, "deterministic": True, "amp": False,
                             "localization_loss_weight": 0.0, "aux_loss_weight": 0.0,
                             "feature_pair_loss_weight": 0.0, "best_metric": "foreground_miou"},
                "inference": {"test_time_augmentation": "none", "prediction_rule": "damage_argmax"},
            })
        config["experiment"] = {
            "id": identifier, "canonical_id": identifier, "dataset": ds, "suite": suite,
            "model_variant": variant, "roles": list(dict.fromkeys(roles)), "uses": list(uses),
            "paper_refs": list(dict.fromkeys(ref for entry in uses for ref in entry["paper_refs"])),
            "protocol": "controlled_comparison_v1", "status": "configured",
            "comparison": {"kind": kind, "reference": reference or "", "factor": factor or suffix,
                           "allowed_changes": allowed_changes},
        }
        config["logging"] = {"run_name": identifier}
        for section, values in overrides.items():
            config.setdefault(section, {}).update(values)
        path = root / "configs" / "experiments" / suite / f"{identifier}.xml"
        xml_write(path, config)
        paths.append(path)

    def baseline(ds, name):
        return {"name": name, "compare_model": name, "compare_adapt_batchnorm": True,
                "compare_deep_supervision": False, "compare_target_channels": 3,
                "encoder_checkpoint": "", "selective_scan_backend": "auto",
                "changemamba_variant": "bda" if ds == "xbd" else "multimodal",
                "share_damage_encoder": ds == "xbd"}

    for ds in DATASETS:
        qualitative_refs = ["Figs.8–11"] if ds in ("bright", "haiti") else ["Fig.13" if ds == "cau_flood" else "Fig.14"]
        for name in ("dehcd_s", "dehcd_m", "dehcd_l", *LEGACY_BASELINES,
                     *(DAMAGE_BASELINES if ds in ("bright", "haiti") else ())):
            variant = name.rsplit("_", 1)[-1] if name.startswith("dehcd_") else name
            roles = ["main", "qualitative", "efficiency"]
            uses = [use("main", "model", [TABLES[ds]]), use(f"main_{ds}", "model", [TABLES[ds]]),
                    use("qualitative", "checkpoint", qualitative_refs, display_seed=42, postprocess_only=True),
                    use("efficiency", "checkpoint", ["Figs.15/16"], display_seed=42, postprocess_only=True)]
            if ds in ("bright", "haiti"):
                roles.append("confusion")
                uses.append(use("confusion", "checkpoint", ["Figs.8–11"], display_seed=42, postprocess_only=True))
            if name.startswith("dehcd_"):
                roles.append("scaling")
                uses.extend([use("scaling", "size", ["Figs.15/16"], x=variant),
                             use(f"scaling_{ds}", "size", ["Figs.15/16"], x=variant)])
            if name == "dehcd_l":
                roles.append("data_statistics")
                uses.append(use("data_statistics", "data", ["Table 1"], postprocess_only=True))
                if ds in ("bright", "haiti"):
                    roles.extend(["ablation", "features"])
                    uses.extend([use("ablation", "reference", ["Fig.17"]),
                                 use(f"fig17_{ds}", "reference", ["Fig.17"], modules=[1, 1, 1, 1]),
                                 use("features", "checkpoint", ["Fig.12"], display_seed=42, postprocess_only=True)])
                if ds == "bright":
                    for group in ("capacity", "dpm", "recipe"):
                        roles.append(group)
                        uses.append(use(group, "reference", [f"Additional {group} comparison"]))
                if ds == "haiti":
                    roles.append("sensitivity")
                    uses.extend([use("sensitivity", "reference", ["Fig.19"]),
                                 use("hog_bins", "default", ["Fig.19"], x=6, reference="haiti_dehcd_l")])
            if ds == "bright" and name == "dehcd_m":
                roles.append("sensitivity")
                uses.extend([use("sensitivity", "reference", ["Figs.18/19"]),
                             use("irb_steps", "default", ["Fig.18"], x=3, reference="bright_dehcd_m"),
                             use("hog_levels", "default", ["Fig.19"], x=2, levels=[0, 1], reference="bright_dehcd_m")])
            model = {"backbone": name, "base_channels": {"s": 16, "m": 24, "l": 32}[variant]} if name.startswith("dehcd_") else baseline(ds, name)
            overrides = {"model": model}
            if not name.startswith("dehcd_"):
                overrides["experiment"] = {"implementation": "adapted_official_core",
                    "adapter": "learned_input_GN_shared_primary_objective"}
            emit(ds, name, "main", overrides, variant=variant, roles=roles, uses=uses,
                 kind="architecture", factor="architecture", allowed_changes=["model"])

    for ds in ("bright", "haiti"):
        for name, switches in COMPONENTS.items():
            modules = [int(switches.get("use_hog", True)), int(switches.get("fusion_mode", "dpm") == "dpm"),
                       int(switches.get("global_context", True) and switches.get("cross_scale_fusion", True)),
                       int(switches.get("diffusion_steps", 3) > 0)]
            emit(ds, name, "ablation", {"model": switches}, roles=["ablation"],
                 uses=[use("ablation", "control", ["Fig.17"]),
                       use(f"fig17_{ds}", "control", ["Fig.17"], modules=modules)], factor="components")
    for group, changes in (("capacity", CAPACITY), ("dpm", DPM)):
        for name, switches in changes.items():
            emit("bright", name, "ablation", {"model": switches}, roles=[group],
                 uses=[use(group, "control", [f"Additional {group} comparison"])])
    simple_loss = {"loss": "compound", "ce_weight": 1.0, "dice_weight": 1.0,
                   "focal_weight": 0.0, "lovasz_weight": 0.0, "tversky_weight": 0.0,
                   "foreground_dice_weight": 0.0, "label_smoothing": 0.0,
                   "aux_loss_weight": 0.0, "feature_pair_loss_weight": 0.0}
    for name, settings, factor in (("ce_dice_full_sampling", simple_loss, "loss_recipe"),
            ("no_weighted_sampler", {"class_balanced_sampler": False}, "weighted_sampler"),
            ("no_class_weights", {"class_weights": []}, "class_weights")):
        emit("bright", name, "recipe", {"training": settings}, roles=["recipe"],
             uses=[use("recipe", "control", ["Additional recipe comparison"])], factor=factor,
             kind="recipe_bundle" if name == "ce_dice_full_sampling" else "control")
    for bins in (2, 4, 8, 10):
        emit("haiti", f"hog_bins_{bins}", "sensitivity", {"model": {"hog_bins": bins}}, roles=["sensitivity"],
             uses=[use("sensitivity", "point", ["Fig.19"]),
                   use("hog_bins", "point", ["Fig.19"], x=bins, reference="haiti_dehcd_l")], factor="hog_bins")
    for steps in (0, 1, 2, 4, 5, 6, 7, 8):
        emit("bright", f"m_irb_steps_{steps}", "sensitivity", {"model": {"diffusion_steps": steps}},
             variant="m", reference="bright_dehcd_m", factor="diffusion_steps", roles=["sensitivity"],
             uses=[use("sensitivity", "point", ["Fig.18"]),
                   use("irb_steps", "point", ["Fig.18"], x=steps, reference="bright_dehcd_m")])
    for levels in (1, 3, 4):
        emit("bright", f"m_hog_levels_{levels}", "sensitivity", {"model": {"hog_modulation_levels": levels}},
             variant="m", reference="bright_dehcd_m", factor="hog_modulation_levels", roles=["sensitivity"],
             uses=[use("sensitivity", "point", ["Fig.19"]),
                   use("hog_levels", "point", ["Fig.19"], x=levels, levels=list(range(levels)), reference="bright_dehcd_m")])
    catalog = derive_catalog(root, paths)
    write_catalog_views(root, catalog, SEEDS)
    return catalog


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(f"Generated {len(generate())} unique definitions under configs/experiments/")
