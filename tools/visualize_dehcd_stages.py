from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_CACHE_DIR = Path("/tmp/matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from tools.select_paper_samples import (
    dataset_key as selection_dataset_key,
    identify_model,
    is_extra_run,
    load_contact_font,
    load_config_for_result,
    normalized_id_variants,
    resize_panel_image,
    result_metric,
    sample_aliases,
)
from tools.test import (
    build_visual_style,
    colorize_mask_array,
    error_map_array,
    sanitize_filename,
    tensor_preview_array,
)


@dataclass
class HacfRun:
    dataset: str
    train_run: str
    train_run_dir: Path
    checkpoint: Path
    split: str
    result_path: Path
    score: float
    metric_name: str
    config: dict[str, Any]


@dataclass
class SampleRequest:
    dataset: str
    sample_id: str
    sample_dir: Path
    rank: int


@dataclass
class Capture:
    name: str
    call_index: int
    tensor_index: int
    phase: str
    tensor: Any
    grad: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready DEHCD-Net(l) intermediate activation maps, probability maps, "
            "and Grad-CAM heatmaps for selected top-k qualitative samples."
        )
    )
    parser.add_argument("--sample-root", default="runs/paper_selected_samples", help="Top-k sample folder produced by select_paper_samples.py.")
    parser.add_argument("--test-root", default="runs/test_paper_all_3", help="Batch test output folder containing *.json run summaries.")
    parser.add_argument("--output-root", default="runs/paper_stage_visuals", help="Where stage visualizations are written.")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint selector fallback when result JSON has no checkpoint path.")
    parser.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"], help="Dataset split. auto uses each test JSON split.")
    parser.add_argument("--include-extra-runs", action="store_true", help="Allow discussion/ablation runs when choosing DEHCD-Net(l) checkpoints.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset folders to process, e.g. BRIGHT CAU Haiti.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per dataset; 0 processes all discovered samples.")
    parser.add_argument(
        "--layers",
        default="auto",
        help="Comma-separated module names to hook, or auto for DEHCD-Net module stages.",
    )
    parser.add_argument(
        "--gradcam-layers",
        default="auto",
        help="Comma-separated module names for Grad-CAM, auto, or none.",
    )
    parser.add_argument(
        "--targets",
        default="foreground",
        help="Grad-CAM/probability targets: foreground, all, or class ids such as 1,2,3.",
    )
    parser.add_argument(
        "--target-mask",
        default="pred",
        choices=["pred", "gt", "union", "all"],
        help="Spatial mask used when reducing segmentation logits for Grad-CAM.",
    )
    parser.add_argument("--max-stage-maps", type=int, default=64, help="Maximum activation maps saved per sample.")
    parser.add_argument("--overlay-alpha", type=float, default=0.48, help="Heatmap overlay opacity.")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for summary contact sheet.")
    parser.add_argument("--dpi", type=int, default=600, help="DPI metadata for saved summary contact sheets.")
    parser.add_argument("--no-summary", action="store_true", help="Skip per-sample summary contact sheet.")
    parser.add_argument("--no-paper-figure", action="store_true", help="Skip per-sample horizontal paper_figure.png outputs.")
    parser.add_argument(
        "--paper-figure-only",
        action="store_true",
        help="Only rebuild per-sample horizontal paper_figure.png files from existing visualization folders.",
    )
    parser.add_argument("--paper-figure-tile-size", type=int, default=640, help="Panel size in pixels for each paper_figure.png cell.")
    parser.add_argument("--paper-figure-font-size", type=int, default=80, help="Bold label font size for paper_figure.png.")
    parser.add_argument(
        "--dataset-paper-grid",
        action="store_true",
        help="Also build the legacy dataset-level multi-sample paper_grid.png.",
    )
    parser.add_argument("--no-paper-grid", action="store_true", help="Deprecated alias for skipping the legacy dataset-level paper_grid.png.")
    parser.add_argument(
        "--paper-grid-only",
        action="store_true",
        help="Only rebuild legacy dataset-level paper_grid.png from existing per-sample visualization folders.",
    )
    parser.add_argument("--paper-grid-samples", type=int, default=5, help="Number of sample columns in dataset-level paper_grid.png.")
    parser.add_argument("--paper-grid-tile-size", type=int, default=170, help="Cell size in pixels for dataset-level paper_grid.png.")
    parser.add_argument(
        "--paper-grid-rows",
        default=(
            "input,hog_optical_before,hog_optical_after,hog_sar_before,hog_sar_after,"
            "dpm_before,dpm_after,bicsf_before,bicsf_after,irb_before,irb_after,head_heatmap,gt"
        ),
        help=(
            "Comma-separated rows for paper figures. Available: input,hog_optical_before,hog_optical_after,"
            "hog_sar_before,hog_sar_after,dpm_before,dpm_after,bicsf_before,gcbm_after,bicsf_after,"
            "irb_before,irb_after,head_heatmap,prob,gradcam,pred,gt. In per-sample figures, GCBM is folded into BiCSF."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample visualization folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sample_root = Path(args.sample_root).expanduser()
    test_root = Path(args.test_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if bool(args.paper_figure_only):
        written = rebuild_existing_sample_paper_figures(
            output_root=output_root,
            datasets=args.datasets,
            row_specs=parse_paper_grid_rows(str(args.paper_grid_rows)),
            tile_size=int(args.paper_figure_tile_size),
            dpi=int(args.dpi),
            font_size=int(args.paper_figure_font_size),
        )
        print(f"Rebuilt {written} per-sample paper figure(s) under {output_root}")
        return
    if bool(args.paper_grid_only):
        written = rebuild_existing_paper_grids(
            output_root=output_root,
            datasets=args.datasets,
            row_specs=parse_paper_grid_rows(str(args.paper_grid_rows)),
            sample_count=int(args.paper_grid_samples),
            tile_size=int(args.paper_grid_tile_size),
            dpi=int(args.dpi),
        )
        print(f"Rebuilt {written} dataset-level paper grid(s) under {output_root}")
        return
    if not sample_root.exists():
        raise FileNotFoundError(f"sample root not found: {sample_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"test root not found: {test_root}")

    device = resolve_device(args.device)
    samples_by_dataset = discover_sample_requests(sample_root, datasets=args.datasets, max_samples=int(args.max_samples))
    runs = discover_dehcd_l_runs(
        test_root,
        checkpoint_selector=str(args.checkpoint),
        split_override=str(args.split),
        include_extra_runs=bool(args.include_extra_runs),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "sample_root": str(sample_root),
        "test_root": str(test_root),
        "output_root": str(output_root),
        "device": str(device),
        "datasets": {},
        "method_note": (
            "Activation maps are channel-mean absolute responses from forward hooks. "
            "Grad-CAM maps use target-class gradients with respect to hooked feature maps; "
            "Module-effect rows use paper terminology: HOG, DPM, BiCSF, GCBM/WASM, and IRB. "
            "No model architecture or weights are modified."
        ),
    }

    for dataset, requests in sorted(samples_by_dataset.items()):
        run = runs.get(dataset)
        if run is None:
            print(f"Warning: no DEHCD-Net(l) test run found for {dataset}; skipped.", file=sys.stderr)
            continue
        print(f"{dataset}: using DEHCD-Net(l) run {run.train_run} ({run.metric_name}={run.score:.4f})")
        processed = process_dataset(dataset, requests, run, args, device, output_root)
        paper_grid = None
        if processed and bool(args.dataset_paper_grid) and not bool(args.no_paper_grid):
            paper_grid = make_dataset_paper_grid(
                dataset_dir=output_root / dataset,
                sample_dirs=[Path(item["output_dir"]) for item in processed if item.get("output_dir")],
                row_specs=parse_paper_grid_rows(str(args.paper_grid_rows)),
                sample_count=int(args.paper_grid_samples),
                tile_size=int(args.paper_grid_tile_size),
                dpi=int(args.dpi),
            )
        summary["datasets"][dataset] = {
            "run": run.train_run,
            "checkpoint": str(run.checkpoint),
            "split": run.split,
            "sample_count": len(processed),
            "paper_grid": str(paper_grid) if paper_grid else None,
            "samples": processed,
        }

    with (output_root / "stage_visualization_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Saved DEHCD-Net(l) stage visualizations to {output_root}")


def resolve_device(value: str):
    import torch

    if str(value).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def discover_sample_requests(sample_root: Path, datasets: Iterable[str] | None, max_samples: int) -> dict[str, list[SampleRequest]]:
    allowed = {item.lower() for item in datasets or []}
    out: dict[str, list[SampleRequest]] = {}
    for dataset_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        if allowed and dataset.lower() not in allowed:
            continue
        requests: list[SampleRequest] = []
        for sample_dir in sorted((path for path in dataset_dir.iterdir() if path.is_dir()), key=sample_dir_sort_key):
            sample_id = sample_id_from_dir(sample_dir)
            rank = sample_rank(sample_dir.name)
            requests.append(SampleRequest(dataset=dataset, sample_id=sample_id, sample_dir=sample_dir, rank=rank))
            if max_samples > 0 and len(requests) >= max_samples:
                break
        if requests:
            out[dataset] = requests
    if not out:
        raise RuntimeError(f"No sample folders found under {sample_root}")
    return out


def sample_dir_sort_key(path: Path) -> tuple[int, str]:
    return sample_rank(path.name), path.name


def sample_rank(name: str) -> int:
    match = re.match(r"top-(\d+)-", name)
    return int(match.group(1)) if match else 10**9


def sample_id_from_dir(sample_dir: Path) -> str:
    meta_path = sample_dir / "sample_selection.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("sample_id"):
            return str(payload["sample_id"])
    return re.sub(r"^top-\d+-", "", sample_dir.name)


def discover_dehcd_l_runs(test_root: Path, checkpoint_selector: str, split_override: str, include_extra_runs: bool = False) -> dict[str, HacfRun]:
    best: dict[str, HacfRun] = {}
    for result_path in sorted(test_root.glob("*.json")):
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        train_run = str(result.get("train_run") or result_path.stem)
        if is_extra_run(train_run) and not include_extra_runs:
            continue
        config = load_config_for_result(result)
        model_key, _display, role, size = identify_model(result, config)
        if role != "dehcd" or size != "l":
            continue
        dataset = selection_dataset_key(result, config)
        metric_name = str(result.get("best_metric") or result.get("checkpoint_best_metric_name") or "primary_score")
        score = result_metric(result, metric_name)
        train_run_dir = Path(str(result.get("train_run_dir") or (Path("runs/train") / train_run))).expanduser()
        checkpoint = resolve_checkpoint_from_result(result, train_run_dir, checkpoint_selector)
        split = str(split_override if split_override != "auto" else result.get("split") or "test")
        run = HacfRun(
            dataset=dataset,
            train_run=train_run,
            train_run_dir=train_run_dir,
            checkpoint=checkpoint,
            split=split,
            result_path=result_path,
            score=score,
            metric_name=metric_name,
            config=config,
        )
        current = best.get(dataset)
        if current is None or run.score > current.score:
            best[dataset] = run
    return best


def resolve_checkpoint_from_result(result: dict[str, Any], train_run_dir: Path, selector: str) -> Path:
    result_checkpoint = Path(str(result.get("checkpoint") or "")).expanduser()
    if result_checkpoint.exists():
        return result_checkpoint
    checkpoint_dir = train_run_dir / "checkpoints"
    selector = str(selector or "best")
    explicit = Path(selector).expanduser()
    if explicit.exists():
        return explicit
    if selector.lower() == "best":
        path = checkpoint_dir / "best.pth"
    elif selector.lower() in {"latest", "last"}:
        path = latest_epoch_checkpoint(checkpoint_dir)
    else:
        path = checkpoint_dir / selector
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found for {train_run_dir}: {path}")
    return path


def latest_epoch_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("epoch_*.pth"), key=checkpoint_epoch)
    if candidates:
        return candidates[-1]
    best = checkpoint_dir / "best.pth"
    if best.exists():
        return best
    raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def process_dataset(
    dataset_name: str,
    requests: list[SampleRequest],
    run: HacfRun,
    args: argparse.Namespace,
    device,
    output_root: Path,
) -> list[dict[str, Any]]:
    import torch
    from datasets import build_dataset
    from models import build_model
    from utils.checkpoint import load_model_state

    config = copy.deepcopy(run.config)
    dataset = build_dataset(config, split=run.split, training=False)
    checkpoint = torch.load(run.checkpoint, map_location=device)
    model = build_model(
        config,
        optical_channels=int(checkpoint.get("optical_channels", dataset.num_optical_channels)),
        sar_channels=int(checkpoint.get("sar_channels", dataset.num_sar_channels)),
    ).to(device)
    load_model_state(model, checkpoint["model"])
    model.eval()

    index = build_sample_index(dataset)
    layer_names = resolve_layer_names(model, str(args.layers))
    gradcam_layer_names = resolve_gradcam_layers(model, str(args.gradcam_layers), layer_names)
    target_specs = parse_target_specs(str(args.targets), num_classes_from_config(config))
    visual_style = build_visual_style(config, num_classes_from_config(config))
    processed: list[dict[str, Any]] = []

    for request in requests:
        idx = find_sample_index(index, request.sample_id)
        if idx is None:
            print(f"Warning: sample not found in {dataset_name}/{run.split}: {request.sample_id}", file=sys.stderr)
            continue
        sample_out = output_root / dataset_name / request.sample_dir.name
        if sample_out.exists() and not bool(args.overwrite):
            print(f"Skip existing {sample_out} (use --overwrite to regenerate).")
            paper_figure = None
            if not bool(args.no_paper_figure):
                paper_figure = make_sample_paper_figure(
                    sample_dir=sample_out,
                    row_specs=parse_paper_grid_rows(str(args.paper_grid_rows)),
                    tile_size=int(args.paper_figure_tile_size),
                    dpi=int(args.dpi),
                    font_size=int(args.paper_figure_font_size),
                )
            processed.append(
                {
                    "sample_id": request.sample_id,
                    "output_dir": str(sample_out),
                    "paper_figure": str(paper_figure) if paper_figure else None,
                    "skipped_existing": True,
                }
            )
            continue
        sample_out.mkdir(parents=True, exist_ok=True)
        item = dataset[idx]
        record = explain_sample(
            model=model,
            item=item,
            request=request,
            output_dir=sample_out,
            layer_names=layer_names,
            gradcam_layer_names=gradcam_layer_names,
            target_specs=target_specs,
            visual_style=visual_style,
            config=config,
            args=args,
            device=device,
        )
        processed.append(record)
    return processed


def build_sample_index(dataset) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, sample in enumerate(getattr(dataset, "samples", [])):
        for alias in sample_aliases(sample):
            for variant in normalized_id_variants(alias):
                out.setdefault(variant, idx)
    return out


def find_sample_index(index: dict[str, int], sample_id: str) -> int | None:
    for variant in normalized_id_variants(sample_id):
        if variant in index:
            return index[variant]
    return None


def num_classes_from_config(config: dict[str, Any]) -> int:
    return int(config.get("task", {}).get("num_classes", config.get("model", {}).get("num_classes", 2)))


def resolve_layer_names(model, value: str) -> list[str]:
    modules = dict(model.named_modules())
    if value.lower() not in {"", "auto", "default"}:
        names = [name for name in split_csv(value) if name in modules]
        for required in module_effect_required_layers():
            if required in modules and required not in names and modules[required].__class__.__name__ != "Identity":
                names.append(required)
        return names

    names: list[str] = []
    append_if_present(names, modules, "hog")
    append_module_list(names, modules, "optical_hog_modulators")
    append_module_list(names, modules, "sar_hog_modulators")
    for idx in range(4):
        append_if_present(names, modules, f"fusion_blocks.{idx}.flow_head")
        append_if_present(names, modules, f"fusion_blocks.{idx}.change_gate")
        append_if_present(names, modules, f"fusion_blocks.{idx}")
    for name in ["global_context", "cross_scale_fusion", "bottleneck", "diffusion_refine", "dec2", "dec1", "dec0", "head.0"]:
        append_if_present(names, modules, name)
    return names


def module_effect_required_layers() -> list[str]:
    return [
        "hog",
        "optical_hog_modulators.0",
        "sar_hog_modulators.0",
        "fusion_blocks.0",
        "global_context",
        "cross_scale_fusion",
        "diffusion_refine",
    ]


def resolve_gradcam_layers(model, value: str, layer_names: list[str]) -> list[str]:
    modules = dict(model.named_modules())
    if value.lower() == "none":
        return []
    if value.lower() not in {"", "auto", "default"}:
        return [name for name in split_csv(value) if name in modules]
    preferred = ["fusion_blocks.3", "bottleneck", "diffusion_refine", "dec2", "dec1", "dec0", "head.0"]
    return [name for name in preferred if name in modules and name in layer_names]


def append_if_present(out: list[str], modules: dict[str, Any], name: str) -> None:
    if name in modules:
        module = modules[name]
        if module.__class__.__name__ != "Identity":
            out.append(name)


def append_module_list(out: list[str], modules: dict[str, Any], prefix: str) -> None:
    idx = 0
    while f"{prefix}.{idx}" in modules:
        out.append(f"{prefix}.{idx}")
        idx += 1


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_target_specs(value: str, num_classes: int) -> list[str]:
    specs: list[str] = []
    for item in split_csv(value):
        lowered = item.lower()
        if lowered == "all":
            specs.extend(f"class_{idx}" for idx in range(1, num_classes))
        elif lowered in {"fg", "foreground"}:
            specs.append("foreground")
        elif lowered.isdigit():
            idx = int(lowered)
            if 0 <= idx < num_classes:
                specs.append(f"class_{idx}")
        elif lowered.startswith("class_"):
            specs.append(lowered)
    return unique_preserve_order(specs or ["foreground"])


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


class ActivationRecorder:
    def __init__(self, model, layer_names: list[str], keep_graph: bool = False):
        self.model = model
        self.layer_names = layer_names
        self.keep_graph = bool(keep_graph)
        self.handles = []
        self.captures: list[Capture] = []
        self.call_counts: dict[str, int] = {}

    def __enter__(self):
        modules = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in modules:
                print(f"Warning: hook layer not found: {name}", file=sys.stderr)
                continue
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        self.captures.clear()
        self.call_counts.clear()

    def clear_grads(self) -> None:
        for capture in self.captures:
            capture.grad = None

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            call_index = self.call_counts.get(name, 0)
            self.call_counts[name] = call_index + 1
            for tensor_index, tensor in enumerate(flatten_4d_tensors(_inputs)):
                captured_tensor = tensor if self.keep_graph else tensor.detach().cpu()
                self.captures.append(
                    Capture(
                        name=name,
                        call_index=call_index,
                        tensor_index=tensor_index,
                        phase="input",
                        tensor=captured_tensor,
                    )
                )
            for tensor_index, tensor in enumerate(flatten_4d_tensors(output)):
                captured_tensor = tensor if self.keep_graph else tensor.detach().cpu()
                capture = Capture(name=name, call_index=call_index, tensor_index=tensor_index, phase="output", tensor=captured_tensor)
                if self.keep_graph and getattr(tensor, "requires_grad", False):
                    tensor.register_hook(lambda grad, cap=capture: set_capture_grad(cap, grad))
                self.captures.append(capture)

        return hook


def set_capture_grad(capture: Capture, grad):
    capture.grad = grad
    return None


def flatten_4d_tensors(output) -> list[Any]:
    import torch

    if isinstance(output, torch.Tensor):
        return [output] if output.ndim == 4 else []
    tensors: list[Any] = []
    if isinstance(output, (list, tuple)):
        for item in output:
            tensors.extend(flatten_4d_tensors(item))
    if isinstance(output, dict):
        for item in output.values():
            tensors.extend(flatten_4d_tensors(item))
    return tensors


def explain_sample(
    model,
    item: dict[str, Any],
    request: SampleRequest,
    output_dir: Path,
    layer_names: list[str],
    gradcam_layer_names: list[str],
    target_specs: list[str],
    visual_style: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
    device,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from utils.model_outputs import extract_logits

    num_classes = int(visual_style["num_classes"])
    ignore_index = int(config.get("task", {}).get("ignore_index", config.get("dataset", {}).get("ignore_index", 255)))
    optical = item["optical"].unsqueeze(0).to(device)
    sar = item["sar"].unsqueeze(0).to(device)
    label = item["label"].to(device).long()
    base_optical = tensor_preview_array(optical, sar=False)
    needs_gradcam = bool(gradcam_layer_names)

    with ActivationRecorder(model, layer_names, keep_graph=needs_gradcam) as recorder:
        recorder.clear()
        if needs_gradcam:
            model.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if needs_gradcam else torch.inference_mode()
        with grad_context:
            logits = extract_logits(model(optical, sar))
            prob = torch.softmax(logits.float(), dim=1)[0]
            pred = torch.argmax(logits, dim=1)[0]

        save_base_outputs(
            output_dir=output_dir,
            optical=optical,
            sar=sar,
            label=label,
            pred=pred,
            prob=prob,
            visual_style=visual_style,
            ignore_index=ignore_index,
            target_specs=target_specs,
            dpi=int(args.dpi),
        )
        activation_records = save_activation_maps(
            captures=recorder.captures,
            output_dir=output_dir / "activations",
            overlay_dir=output_dir / "activation_overlays",
            base_rgb=base_optical,
            input_size=label.shape[-2:],
            max_maps=int(args.max_stage_maps),
            alpha=float(args.overlay_alpha),
        )
        module_effect_records = save_module_effect_maps(
            captures=recorder.captures,
            output_dir=output_dir / "module_effects",
            overlay_dir=output_dir / "module_effect_overlays",
            base_rgb=base_optical,
            input_size=label.shape[-2:],
            alpha=float(args.overlay_alpha),
            head_heatmap=probability_map(prob, target_specs[0] if target_specs else "foreground"),
        )

        gradcam_records: list[dict[str, Any]] = []
        if needs_gradcam:
            capture_by_layer = {capture_display_name(capture): capture for capture in recorder.captures}
            for target_spec in target_specs:
                recorder.clear_grads()
                model.zero_grad(set_to_none=True)
                target = segmentation_target_score(
                    logits=logits,
                    pred=pred,
                    label=label,
                    spec=target_spec,
                    mask_mode=str(args.target_mask),
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
                target.backward(retain_graph=True)
                for layer_name in gradcam_layer_names:
                    for display_name, capture in capture_by_layer.items():
                        if capture.name != layer_name:
                            continue
                        cam = gradcam_from_capture(capture)
                        if cam is None:
                            continue
                        cam = F.interpolate(cam[None, None], size=label.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                        cam_np = normalize_tensor_map(cam)
                        safe_target = sanitize_filename(target_spec)
                        safe_layer = sanitize_filename(display_name)
                        heat_path = output_dir / "gradcam" / safe_target / f"{safe_layer}.png"
                        overlay_path = output_dir / "gradcam_overlays" / safe_target / f"{safe_layer}.png"
                        save_heatmap_image(cam_np, heat_path)
                        save_overlay(base_optical, cam_np, overlay_path, alpha=float(args.overlay_alpha))
                        gradcam_records.append({"target": target_spec, "layer": display_name, "heatmap": str(heat_path), "overlay": str(overlay_path)})
        del logits, prob, pred
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not bool(args.no_summary):
        make_summary_contact_sheet(
            output_dir,
            tile_size=int(args.tile_size),
            dpi=int(args.dpi),
            font_size=int(args.paper_figure_font_size),
        )

    paper_figure = None
    if not bool(args.no_paper_figure):
        paper_figure = make_sample_paper_figure(
            sample_dir=output_dir,
            row_specs=parse_paper_grid_rows(str(args.paper_grid_rows)),
            tile_size=int(args.paper_figure_tile_size),
            dpi=int(args.dpi),
            font_size=int(args.paper_figure_font_size),
        )

    record = {
        "dataset": request.dataset,
        "rank": request.rank,
        "sample_id": request.sample_id,
        "output_dir": str(output_dir),
        "layers": layer_names,
        "gradcam_layers": gradcam_layer_names,
        "targets": target_specs,
        "activation_maps": activation_records,
        "module_effect_maps": module_effect_records,
        "gradcam_maps": gradcam_records,
        "paper_figure": str(paper_figure) if paper_figure else None,
    }
    with (output_dir / "stage_visualization_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return record


def save_base_outputs(
    output_dir: Path,
    optical,
    sar,
    label,
    pred,
    prob,
    visual_style: dict[str, Any],
    ignore_index: int,
    target_specs: list[str],
    dpi: int,
) -> None:
    import torch

    num_classes = int(visual_style["num_classes"])
    class_colors = visual_style["class_colors"]
    save_rgb_array(tensor_preview_array(optical, sar=False), output_dir / "inputs" / "pre_optical.png")
    save_rgb_array(tensor_preview_array(sar, sar=True), output_dir / "inputs" / "post_sar.png")
    save_rgb_array(colorize_mask_array(label, num_classes=num_classes, ignore_index=ignore_index, class_colors=class_colors), output_dir / "masks" / "gt.png")
    save_rgb_array(colorize_mask_array(pred, num_classes=num_classes, ignore_index=ignore_index, class_colors=class_colors), output_dir / "masks" / "prediction.png")
    save_rgb_array(error_map_array(pred, label, ignore_index=ignore_index), output_dir / "masks" / "error_map.png")
    for spec in target_specs:
        heatmap = probability_map(prob, spec)
        save_heatmap_image(heatmap, output_dir / "probabilities" / f"{sanitize_filename(spec)}.png")


def probability_map(prob, spec: str):
    if spec == "foreground":
        heatmap = prob[1:].sum(dim=0) if prob.shape[0] > 1 else prob[0]
    elif spec.startswith("class_"):
        idx = int(spec.split("_", 1)[1])
        heatmap = prob[idx]
    else:
        heatmap = prob[0]
    return normalize_tensor_map(heatmap)


def save_activation_maps(
    captures: list[Capture],
    output_dir: Path,
    overlay_dir: Path,
    base_rgb,
    input_size: tuple[int, int],
    max_maps: int,
    alpha: float,
) -> list[dict[str, Any]]:
    import torch.nn.functional as F

    records: list[dict[str, Any]] = []
    output_captures = [capture for capture in captures if capture.phase == "output"]
    for idx, capture in enumerate(output_captures[: max(max_maps, 0)]):
        name = capture_display_name(capture)
        tensor = capture.tensor.detach()
        act = activation_heatmap(tensor, name=name)
        act = F.interpolate(act[None, None], size=input_size, mode="bilinear", align_corners=False)[0, 0]
        act_np = normalize_tensor_map(act)
        safe = f"{idx:02d}_{sanitize_filename(name)}"
        heat_path = output_dir / f"{safe}.png"
        overlay_path = overlay_dir / f"{safe}.png"
        save_heatmap_image(act_np, heat_path)
        save_overlay(base_rgb, act_np, overlay_path, alpha=alpha)
        records.append({"layer": name, "heatmap": str(heat_path), "overlay": str(overlay_path)})
    return records


def activation_heatmap(tensor, name: str):
    import torch

    x = tensor[0].float()
    if x.shape[0] == 2 and "flow_head" in name:
        return torch.sqrt(x[0].square() + x[1].square())
    return x.abs().mean(dim=0)


def save_module_effect_maps(
    captures: list[Capture],
    output_dir: Path,
    overlay_dir: Path,
    base_rgb,
    input_size: tuple[int, int],
    alpha: float,
    head_heatmap=None,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    specs: list[tuple[str, str, Any]] = []

    optical_hog_before = find_capture(captures, "optical_hog_modulators.0", phase="input", tensor_index=0)
    optical_hog_after = find_capture(captures, "optical_hog_modulators.0", phase="output", tensor_index=0)
    sar_hog_before = find_capture(captures, "sar_hog_modulators.0", phase="input", tensor_index=0)
    sar_hog_after = find_capture(captures, "sar_hog_modulators.0", phase="output", tensor_index=0)
    optical_hog_prior = find_capture(captures, "hog", phase="output", call_index=0)
    sar_hog_prior = find_capture(captures, "hog", phase="output", call_index=1)
    if optical_hog_prior is not None:
        specs.append(("hog_optical_prior", "Optical HOG prior", activation_heatmap(optical_hog_prior.tensor.detach(), "hog_optical_prior")))
        specs.append(("hog_prior", "HOG prior", activation_heatmap(optical_hog_prior.tensor.detach(), "hog_prior")))
    if sar_hog_prior is not None:
        specs.append(("hog_sar_prior", "SAR HOG prior", activation_heatmap(sar_hog_prior.tensor.detach(), "hog_sar_prior")))
    if optical_hog_before is not None:
        optical_before_heat = activation_heatmap(optical_hog_before.tensor.detach(), "hog_optical_before")
        specs.append(("hog_optical_before", "Optical before HOG", optical_before_heat))
        specs.append(("hog_before", "Before HOG", optical_before_heat))
    if optical_hog_after is not None:
        optical_after_heat = activation_heatmap(optical_hog_after.tensor.detach(), "hog_optical_after")
        specs.append(("hog_optical_after", "Optical after HOG", optical_after_heat))
        specs.append(("hog_after", "After HOG", optical_after_heat))
    if sar_hog_before is not None:
        specs.append(("hog_sar_before", "SAR before HOG", activation_heatmap(sar_hog_before.tensor.detach(), "hog_sar_before")))
    if sar_hog_after is not None:
        specs.append(("hog_sar_after", "SAR after HOG", activation_heatmap(sar_hog_after.tensor.detach(), "hog_sar_after")))

    dpm_in0 = find_capture(captures, "fusion_blocks.0", phase="input", tensor_index=0)
    dpm_in1 = find_capture(captures, "fusion_blocks.0", phase="input", tensor_index=1)
    dpm_after = find_capture(captures, "fusion_blocks.0", phase="output", tensor_index=0)
    if dpm_in0 is not None and dpm_in1 is not None:
        specs.append(("dpm_before", "Before DPM", paired_difference_heatmap(dpm_in0.tensor.detach(), dpm_in1.tensor.detach())))
    if dpm_after is not None:
        specs.append(("dpm_after", "After DPM", activation_heatmap(dpm_after.tensor.detach(), "dpm_after")))

    bicsf_before = dpm_after
    gcbm_after = find_capture(captures, "global_context", phase="output", tensor_index=0)
    bicsf_after = find_capture(captures, "cross_scale_fusion", phase="output", tensor_index=0)
    if bicsf_before is not None:
        specs.append(("bicsf_before", "Before BiCSF", activation_heatmap(bicsf_before.tensor.detach(), "bicsf_before")))
    if gcbm_after is not None:
        specs.append(("gcbm_after", "After GCBM", activation_heatmap(gcbm_after.tensor.detach(), "gcbm_after")))
    if bicsf_after is not None:
        specs.append(("bicsf_after", "After BiCSF", activation_heatmap(bicsf_after.tensor.detach(), "bicsf_after")))

    irb_before = find_capture(captures, "diffusion_refine", phase="input", tensor_index=0)
    irb_after = find_capture(captures, "diffusion_refine", phase="output", tensor_index=0)
    if irb_before is not None:
        specs.append(("irb_before", "Before IRB", activation_heatmap(irb_before.tensor.detach(), "irb_before")))
    if irb_after is not None:
        specs.append(("irb_after", "After IRB", activation_heatmap(irb_after.tensor.detach(), "irb_after")))
    if head_heatmap is not None:
        specs.append(("head_heatmap", "Head heatmap", torch.as_tensor(head_heatmap).float()))

    records: list[dict[str, Any]] = []
    for key, label, heat in specs:
        if heat.ndim != 2:
            continue
        heat = torch.as_tensor(heat).float()
        heat = F.interpolate(heat[None, None], size=input_size, mode="bilinear", align_corners=False)[0, 0]
        heat_np = normalize_tensor_map(heat)
        heat_path = output_dir / f"{key}.png"
        overlay_path = overlay_dir / f"{key}.png"
        save_heatmap_image(heat_np, heat_path)
        save_overlay(base_rgb, heat_np, overlay_path, alpha=alpha)
        records.append({"key": key, "label": label, "heatmap": str(heat_path), "overlay": str(overlay_path)})
    return records


def find_capture(
    captures: list[Capture],
    name: str,
    phase: str,
    tensor_index: int | None = None,
    call_index: int | None = None,
) -> Capture | None:
    for capture in captures:
        if capture.name != name or capture.phase != phase:
            continue
        if tensor_index is not None and capture.tensor_index != tensor_index:
            continue
        if call_index is not None and capture.call_index != call_index:
            continue
        return capture
    return None


def paired_difference_heatmap(first, second):
    import torch.nn.functional as F

    x = first[0].float()
    y = second[0].float()
    if x.shape[-2:] != y.shape[-2:]:
        y = F.interpolate(y[None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
    channels = min(x.shape[0], y.shape[0])
    return (x[:channels] - y[:channels]).abs().mean(dim=0)


def capture_display_name(capture: Capture) -> str:
    if capture.name == "hog":
        base = "hog_optical" if capture.call_index == 0 else "hog_sar"
    else:
        base = capture.name
        if capture.call_index > 0:
            base = f"{base}.call{capture.call_index}"
    if capture.tensor_index > 0:
        base = f"{base}.s{capture.tensor_index}"
    if capture.phase != "output":
        base = f"{base}.{capture.phase}{capture.tensor_index}"
    return base


def segmentation_target_score(logits, pred, label, spec: str, mask_mode: str, num_classes: int, ignore_index: int):
    import torch

    if spec == "foreground":
        score_map = logits[0, 1:].amax(dim=0) if num_classes > 2 else logits[0, min(1, logits.shape[1] - 1)]
        pred_mask = pred > 0
        gt_mask = label > 0
    elif spec.startswith("class_"):
        class_idx = int(spec.split("_", 1)[1])
        score_map = logits[0, class_idx]
        pred_mask = pred == class_idx
        gt_mask = label == class_idx
    else:
        score_map = logits[0].amax(dim=0)
        pred_mask = torch.ones_like(pred, dtype=torch.bool)
        gt_mask = pred_mask

    valid = label != int(ignore_index)
    if mask_mode == "pred":
        mask = pred_mask & valid
    elif mask_mode == "gt":
        mask = gt_mask & valid
    elif mask_mode == "union":
        mask = (pred_mask | gt_mask) & valid
    else:
        mask = valid
    if not bool(mask.any()):
        mask = valid
    if not bool(mask.any()):
        mask = torch.ones_like(score_map, dtype=torch.bool)
    weights = mask.float()
    return (score_map * weights).sum() / weights.sum().clamp(min=1.0)


def gradcam_from_capture(capture: Capture):
    import torch

    if capture.grad is None:
        return None
    activation = capture.tensor[0].float()
    gradient = capture.grad[0].float()
    if activation.ndim != 3 or gradient.shape != activation.shape:
        return None
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=0))
    return cam


def normalize_tensor_map(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().float().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.zeros_like(arr, dtype=np.float32)
    low, high = np.percentile(arr[finite], [1, 99])
    if float(high - low) < 1e-8:
        low, high = float(arr[finite].min()), float(arr[finite].max())
    return np.clip((arr - low) / max(float(high - low), 1e-8), 0.0, 1.0).astype("float32")


def save_heatmap_image(heatmap, path: Path, cmap: str = "jet") -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = colorize_heatmap(heatmap, cmap=cmap)
    Image.fromarray(rgb).save(path)


def save_overlay(base_rgb, heatmap, path: Path, alpha: float) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.asarray(base_rgb)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    if base.dtype != np.uint8:
        base = (np.clip(base, 0.0, 1.0) * 255).round().astype("uint8")
    heat = colorize_heatmap(heatmap, cmap="jet")
    if heat.shape[:2] != base.shape[:2]:
        heat = np.asarray(Image.fromarray(heat).resize((base.shape[1], base.shape[0]), resample=Image.Resampling.BILINEAR))
    overlay = ((1.0 - alpha) * base.astype("float32") + alpha * heat.astype("float32")).clip(0, 255).astype("uint8")
    Image.fromarray(overlay).save(path)


def colorize_heatmap(heatmap, cmap: str = "magma"):
    import numpy as np

    arr = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    try:
        import matplotlib.colormaps as colormaps

        rgb = colormaps.get_cmap(cmap)(arr)[..., :3]
    except Exception:
        rgb = np.stack([arr, np.sqrt(arr), 1.0 - arr], axis=-1)
    return (rgb * 255).round().astype("uint8")


def save_rgb_array(array, path: Path) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def make_summary_contact_sheet(output_dir: Path, tile_size: int, dpi: int, font_size: int) -> None:
    from PIL import Image, ImageDraw

    entries = [
        ("Pre-event Optical", output_dir / "inputs" / "pre_optical.png"),
        ("Post-event SAR", output_dir / "inputs" / "post_sar.png"),
        ("GT", output_dir / "masks" / "gt.png"),
        ("Prediction", output_dir / "masks" / "prediction.png"),
        ("Error", output_dir / "masks" / "error_map.png"),
    ]
    entries.extend(first_n_named(output_dir / "probabilities", 2, prefix="Prob"))
    entries.extend(first_n_named(output_dir / "module_effects", 6, prefix="Module"))
    entries.extend(first_n_named(output_dir / "activation_overlays", 5, prefix="Act"))
    gradcam_root = output_dir / "gradcam_overlays"
    for target_dir in sorted(path for path in gradcam_root.glob("*") if path.is_dir()):
        entries.extend(first_n_named(target_dir, 3, prefix=f"Grad-CAM {target_dir.name}"))
        break
    entries = [(label, path) for label, path in entries if path.exists()]
    if not entries:
        return

    font = load_contact_font(font_size=max(int(font_size), 10), bold=True)
    gap = max(tile_size // 32, 14)
    label_h = max(int(font_size) + 24, 54)
    margin = max(tile_size // 28, 18)
    cols = min(5, len(entries))
    rows = math.ceil(len(entries) / cols)
    sheet = Image.new(
        "RGB",
        (cols * tile_size + (cols - 1) * gap + 2 * margin, rows * (tile_size + label_h) + (rows - 1) * gap + 2 * margin),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for idx, (label, path) in enumerate(entries):
        image = Image.open(path).convert("RGB")
        image = resize_for_tile(image, tile_size)
        col, row = idx % cols, idx // cols
        x = margin + col * (tile_size + gap)
        y = margin + row * (tile_size + label_h + gap)
        draw_centered_text(draw, (x, y, x + tile_size, y + label_h), label, font)
        sheet.paste(image, (x, y + label_h))
        draw.rectangle((x, y + label_h, x + tile_size - 1, y + label_h + tile_size - 1), outline=(185, 185, 185), width=2)
    sheet.save(output_dir / "stage_summary.png", dpi=(dpi, dpi))


@dataclass(frozen=True)
class PaperGridRow:
    key: str
    label: str


PAPER_GRID_ROW_LABELS = {
    "input": "T1 / T2 image",
    "hog_prior": "HOG prior",
    "hog_optical_prior": "Optical HOG prior",
    "hog_sar_prior": "SAR HOG prior",
    "hog_before": "Before HOG",
    "hog_after": "After HOG",
    "hog_optical_before": "Optical before HOG",
    "hog_optical_after": "Optical after HOG",
    "hog_sar_before": "SAR before HOG",
    "hog_sar_after": "SAR after HOG",
    "dpm_before": "Before DPM",
    "dpm_after": "After DPM",
    "bicsf_before": "Before BiCSF",
    "gcbm_after": "After GCBM",
    "bicsf_after": "After BiCSF",
    "irb_before": "Before IRB",
    "irb_after": "After IRB",
    "head_heatmap": "Head heatmap",
    "hog": "HOG feature",
    "hfb1": "DPM1 feature",
    "hfb2": "DPM2 feature",
    "hfb3": "DPM3 feature",
    "hfb4": "DPM4 feature",
    "csf": "BiCSF feature",
    "bottleneck": "Bottleneck feature",
    "diffusion": "IRB feature",
    "decoder": "Decoder feature",
    "prob": "Probability",
    "gradcam": "Grad-CAM",
    "pred": "Prediction",
    "gt": "Ground truth",
}


def parse_paper_grid_rows(value: str) -> list[PaperGridRow]:
    rows: list[PaperGridRow] = []
    for key in split_csv(value):
        key = key.lower()
        if key not in PAPER_GRID_ROW_LABELS:
            print(f"Warning: unknown paper-grid row '{key}' ignored.", file=sys.stderr)
            continue
        rows.append(PaperGridRow(key=key, label=PAPER_GRID_ROW_LABELS[key]))
    return rows or [PaperGridRow("input", PAPER_GRID_ROW_LABELS["input"]), PaperGridRow("gt", PAPER_GRID_ROW_LABELS["gt"])]


def rebuild_existing_paper_grids(
    output_root: Path,
    datasets: Iterable[str] | None,
    row_specs: list[PaperGridRow],
    sample_count: int,
    tile_size: int,
    dpi: int,
) -> int:
    if not output_root.exists():
        raise FileNotFoundError(f"output root not found: {output_root}")
    allowed = {item.lower() for item in datasets or []}
    written = 0
    for dataset_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if allowed and dataset_dir.name.lower() not in allowed:
            continue
        sample_dirs = sorted((path for path in dataset_dir.iterdir() if path.is_dir()), key=sample_dir_sort_key)
        if not sample_dirs:
            continue
        if make_dataset_paper_grid(dataset_dir, sample_dirs, row_specs=row_specs, sample_count=sample_count, tile_size=tile_size, dpi=dpi):
            written += 1
    return written


def rebuild_existing_sample_paper_figures(
    output_root: Path,
    datasets: Iterable[str] | None,
    row_specs: list[PaperGridRow],
    tile_size: int,
    dpi: int,
    font_size: int,
) -> int:
    if not output_root.exists():
        raise FileNotFoundError(f"output root not found: {output_root}")
    allowed = {item.lower() for item in datasets or []}
    written = 0
    for dataset_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if allowed and dataset_dir.name.lower() not in allowed:
            continue
        for sample_dir in sorted((path for path in dataset_dir.iterdir() if path.is_dir()), key=sample_dir_sort_key):
            if make_sample_paper_figure(
                sample_dir=sample_dir,
                row_specs=row_specs,
                tile_size=tile_size,
                dpi=dpi,
                font_size=font_size,
            ):
                written += 1
    return written


@dataclass(frozen=True)
class PaperFigurePanel:
    label: str
    path: Path | None
    kind: str


@dataclass(frozen=True)
class PaperFigureColumn:
    top: PaperFigurePanel
    bottom: PaperFigurePanel


def make_sample_paper_figure(
    sample_dir: Path,
    row_specs: list[PaperGridRow],
    tile_size: int,
    dpi: int,
    font_size: int,
) -> Path | None:
    from PIL import Image, ImageDraw

    columns = paper_figure_columns(sample_dir, row_specs)
    if not columns:
        return None

    tile_size = max(int(tile_size), 128)
    font_size = max(int(font_size), 10)
    gap = max(tile_size // 28, 18)
    row_gap = max(tile_size // 28, 18)
    margin = max(tile_size // 24, 22)
    label_h = max(font_size + 24, 54)
    tile_w = tile_h = tile_size
    has_heatmap = any(panel.kind == "heatmap" and panel.path is not None for column in columns for panel in (column.top, column.bottom))
    colorbar_w = max(tile_size // 5, 128) if has_heatmap else 0
    colorbar_gap = max(tile_size // 18, 18) if has_heatmap else 0

    sheet_w = len(columns) * tile_w + (len(columns) - 1) * gap + colorbar_gap + colorbar_w + 2 * margin
    sheet_h = 2 * (tile_h + label_h) + row_gap + 2 * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = load_contact_font(font_size=font_size, bold=True)

    for col_idx, column in enumerate(columns):
        x = margin + col_idx * (tile_w + gap)
        for row_idx, panel_info in enumerate((column.top, column.bottom)):
            y = margin + row_idx * (tile_h + label_h + row_gap)
            draw_centered_text(draw, (x, y, x + tile_w, y + label_h), panel_info.label, font=font)
            image_y = y + label_h
            panel = render_paper_figure_panel(panel_info, size=(tile_w, tile_h))
            sheet.paste(panel, (x, image_y))
            draw.rectangle((x, image_y, x + tile_w - 1, image_y + tile_h - 1), outline=(185, 185, 185), width=2)

    if has_heatmap:
        colorbar_x = margin + len(columns) * tile_w + (len(columns) - 1) * gap + colorbar_gap
        colorbar_y = margin + label_h
        colorbar_font = load_contact_font(font_size=max(font_size // 2, 24), bold=True)
        colorbar_h = 2 * tile_h + label_h + row_gap
        paste_colorbar(sheet, draw, x=colorbar_x, y=colorbar_y, width=colorbar_w, height=colorbar_h, font=colorbar_font)

    out_path = sample_dir / "paper_figure.png"
    sheet.save(out_path, dpi=(dpi, dpi))
    return out_path


def paper_figure_columns(sample_dir: Path, row_specs: list[PaperGridRow]) -> list[PaperFigureColumn]:
    requested = {spec.key for spec in row_specs}
    columns: list[PaperFigureColumn] = []

    if "input" in requested:
        columns.append(
            PaperFigureColumn(
                top=PaperFigurePanel("T1 image", existing_path(sample_dir / "inputs" / "pre_optical.png"), "image"),
                bottom=PaperFigurePanel("T2 image", existing_path(sample_dir / "inputs" / "post_sar.png"), "image"),
            )
        )

    hog_requested = bool(
        requested
        & {
            "hog_prior",
            "hog_before",
            "hog_after",
            "hog_optical_prior",
            "hog_optical_before",
            "hog_optical_after",
            "hog_sar_prior",
            "hog_sar_before",
            "hog_sar_after",
            "hog",
        }
    )
    if hog_requested:
        optical_before = paper_grid_cell_source(sample_dir, "hog_optical_before")
        optical_after = paper_grid_cell_source(sample_dir, "hog_optical_after")
        sar_before = paper_grid_cell_source(sample_dir, "hog_sar_before")
        sar_after = paper_grid_cell_source(sample_dir, "hog_sar_after")
        if isinstance(optical_before, Path) or isinstance(optical_after, Path):
            columns.append(
                PaperFigureColumn(
                    top=PaperFigurePanel("Opt-HOG before", optical_before if isinstance(optical_before, Path) else None, "heatmap"),
                    bottom=PaperFigurePanel("Opt-HOG after", optical_after if isinstance(optical_after, Path) else None, "heatmap"),
                )
            )
        if isinstance(sar_before, Path) or isinstance(sar_after, Path):
            columns.append(
                PaperFigureColumn(
                    top=PaperFigurePanel("SAR HOG before", sar_before if isinstance(sar_before, Path) else None, "heatmap"),
                    bottom=PaperFigurePanel("SAR HOG after", sar_after if isinstance(sar_after, Path) else None, "heatmap"),
                )
            )

    module_pairs = [
        ("DPM", "dpm_before", "dpm_after"),
        ("BiCSF", "bicsf_before", "bicsf_after"),
        ("IRB", "irb_before", "irb_after"),
    ]
    for module_name, before_key, after_key in module_pairs:
        if before_key not in requested and after_key not in requested:
            continue
        before_path = paper_grid_cell_source(sample_dir, before_key)
        after_path = paper_grid_cell_source(sample_dir, after_key)
        before = before_path if isinstance(before_path, Path) else None
        after = after_path if isinstance(after_path, Path) else None
        if before is None and after is None:
            continue
        columns.append(
            PaperFigureColumn(
                top=PaperFigurePanel(f"Before {module_name}", before, "heatmap"),
                bottom=PaperFigurePanel(f"After {module_name}", after, "heatmap"),
            )
        )

    head = paper_grid_cell_source(sample_dir, "head_heatmap")
    if not isinstance(head, Path):
        head = paper_grid_cell_source(sample_dir, "prob")
    head = head if isinstance(head, Path) else None
    gt = existing_path(sample_dir / "masks" / "gt.png")
    if head is not None or gt is not None:
        columns.append(
            PaperFigureColumn(
                top=PaperFigurePanel("Head heatmap", head, "heatmap"),
                bottom=PaperFigurePanel("Ground truth", gt, "mask"),
            )
        )
    return [column for column in columns if column.top.path is not None or column.bottom.path is not None]


def render_paper_figure_panel(panel: PaperFigurePanel, size: tuple[int, int]):
    from PIL import Image

    if panel.path is None:
        return Image.new("RGB", size, color=(255, 255, 255))
    image = Image.open(panel.path).convert("RGB")
    return resize_panel_image(image, size=size, categorical=panel.kind == "mask")


def make_dataset_paper_grid(
    dataset_dir: Path,
    sample_dirs: list[Path],
    row_specs: list[PaperGridRow],
    sample_count: int,
    tile_size: int,
    dpi: int,
) -> Path | None:
    from PIL import Image, ImageDraw

    sample_dirs = [path for path in sample_dirs if path.exists()]
    sample_dirs = sorted(sample_dirs, key=sample_dir_sort_key)[: max(int(sample_count), 1)]
    if not sample_dirs:
        return None

    resolved_rows: list[tuple[PaperGridRow, list[Path | tuple[Path, Path] | None]]] = []
    for spec in row_specs:
        cells = [paper_grid_cell_source(sample_dir, spec.key) for sample_dir in sample_dirs]
        if spec.key not in {"input", "gt"} and all(cell is None for cell in cells):
            continue
        resolved_rows.append((spec, cells))
    if not resolved_rows:
        return None

    tile_size = max(int(tile_size), 80)
    label_w = max(int(tile_size * 1.15), 150)
    row_gap = max(tile_size // 18, 8)
    col_gap = max(tile_size // 20, 8)
    margin = max(tile_size // 8, 18)
    bottom_label_h = max(tile_size // 5, 28)
    colorbar_w = max(tile_size // 5, 28)
    colorbar_gap = max(tile_size // 8, 14)
    width = label_w + len(sample_dirs) * tile_size + (len(sample_dirs) - 1) * col_gap + colorbar_gap + colorbar_w + 2 * margin
    height = len(resolved_rows) * tile_size + (len(resolved_rows) - 1) * row_gap + bottom_label_h + 2 * margin

    sheet = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    label_font = load_serif_font(max(tile_size // 11, 14))
    sample_font = load_serif_font(max(tile_size // 12, 13))

    grid_x = margin + label_w
    grid_y = margin
    for row_idx, (spec, cells) in enumerate(resolved_rows):
        y = grid_y + row_idx * (tile_size + row_gap)
        draw.text((margin, y + tile_size // 2 - max(tile_size // 18, 8)), spec.label, fill=(20, 20, 20), font=label_font)
        for col_idx, source in enumerate(cells):
            x = grid_x + col_idx * (tile_size + col_gap)
            image = render_paper_grid_cell(source, spec.key, tile_size)
            sheet.paste(image, (x, y))
            draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline=(225, 225, 225), width=1)

    label_y = grid_y + len(resolved_rows) * tile_size + (len(resolved_rows) - 1) * row_gap + max(bottom_label_h // 5, 4)
    for col_idx, sample_dir in enumerate(sample_dirs):
        x = grid_x + col_idx * (tile_size + col_gap)
        text = f"({chr(ord('a') + col_idx)})"
        draw_centered_text(draw, (x, label_y, x + tile_size, label_y + bottom_label_h), text, sample_font)

    colorbar_x = grid_x + len(sample_dirs) * tile_size + (len(sample_dirs) - 1) * col_gap + colorbar_gap
    colorbar_y = grid_y
    colorbar_h = min(len(resolved_rows) * tile_size + (len(resolved_rows) - 1) * row_gap, max(tile_size * 4, tile_size))
    paste_colorbar(sheet, draw, x=colorbar_x, y=colorbar_y, width=colorbar_w, height=colorbar_h, font=sample_font)

    out_path = dataset_dir / "paper_grid.png"
    sheet.save(out_path, dpi=(dpi, dpi))
    return out_path


def paper_grid_cell_source(sample_dir: Path, key: str):
    module_effect_keys = {
        "hog_prior",
        "hog_optical_prior",
        "hog_sar_prior",
        "hog_before",
        "hog_after",
        "hog_optical_before",
        "hog_optical_after",
        "hog_sar_before",
        "hog_sar_after",
        "dpm_before",
        "dpm_after",
        "bicsf_before",
        "gcbm_after",
        "bicsf_after",
        "irb_before",
        "irb_after",
        "head_heatmap",
    }
    if key in module_effect_keys:
        return existing_path(sample_dir / "module_effects" / f"{key}.png")
    if key == "input":
        pre = sample_dir / "inputs" / "pre_optical.png"
        post = sample_dir / "inputs" / "post_sar.png"
        return (pre, post) if pre.exists() and post.exists() else None
    if key == "gt":
        return existing_path(sample_dir / "masks" / "gt.png")
    if key == "pred":
        return existing_path(sample_dir / "masks" / "prediction.png")
    if key == "prob":
        return first_existing([sample_dir / "probabilities" / "foreground.png", *sorted((sample_dir / "probabilities").glob("*.png"))])
    if key == "gradcam":
        return first_existing([sample_dir / "gradcam" / "foreground" / "dec0.png", *sorted((sample_dir / "gradcam").glob("*/*.png"))])
    patterns = {
        "hog": ["*hog_optical.png", "*hog_sar.png"],
        "hfb1": ["*fusion_blocks.0.png"],
        "hfb2": ["*fusion_blocks.1.png"],
        "hfb3": ["*fusion_blocks.2.png"],
        "hfb4": ["*fusion_blocks.3.png"],
        "csf": ["*cross_scale_fusion*.png"],
        "bottleneck": ["*bottleneck*.png"],
        "diffusion": ["*diffusion_refine*.png"],
        "decoder": ["*dec0.png", "*dec1.png", "*dec2.png"],
    }
    for pattern in patterns.get(key, []):
        matches = sorted((sample_dir / "activations").glob(pattern))
        if matches:
            return matches[0]
    return None


def existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def render_paper_grid_cell(source, key: str, tile_size: int):
    from PIL import Image

    if source is None:
        return Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
    if isinstance(source, tuple):
        cell = Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
        half = tile_size // 2
        pre = resize_panel_image(Image.open(source[0]).convert("RGB"), size=(half, tile_size), categorical=False)
        post = resize_panel_image(Image.open(source[1]).convert("RGB"), size=(tile_size - half, tile_size), categorical=False)
        cell.paste(pre, (0, 0))
        cell.paste(post, (half, 0))
        return cell
    image = Image.open(source).convert("RGB")
    resample = Image.Resampling.NEAREST if key in {"gt", "pred"} else Image.Resampling.LANCZOS
    return resize_for_tile(image, tile_size, resample=resample)


def paste_colorbar(sheet, draw, x: int, y: int, width: int, height: int, font) -> None:
    from PIL import Image
    import numpy as np

    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    bar = colorize_heatmap(np.repeat(gradient, max(width // 2, 8), axis=1), cmap="jet")
    bar_w = max(min(width // 3, width - 42), 12)
    bar_img = Image.fromarray(bar).resize((bar_w, height), resample=Image.Resampling.BILINEAR)
    sheet.paste(bar_img, (x, y))
    tick_x0 = x + bar_img.width + 5
    text_x = tick_x0 + 9
    for value in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        tick_y = y + int((1.0 - value) * height)
        draw.line((tick_x0, tick_y, tick_x0 + 7, tick_y), fill=(20, 20, 20), width=1)
        text = f"{value:.1f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        text_y = min(max(tick_y - text_h // 2, y), y + height - text_h)
        draw.text((text_x, text_y), text, fill=(20, 20, 20), font=font)


def first_n_named(root: Path, n: int, prefix: str) -> list[tuple[str, Path]]:
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*.png"))[:n]:
        out.append((f"{prefix}: {pretty_layer_label(path.stem)}", path))
    return out


def pretty_layer_label(text: str) -> str:
    text = re.sub(r"^\d+_", "", text)
    return text.replace("_", ".")[:32]


def resize_for_tile(image, tile_size: int, resample=None):
    from PIL import Image, ImageOps

    resample = resample or Image.Resampling.LANCZOS
    image = ImageOps.contain(image, (tile_size, tile_size), method=resample)
    canvas = Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
    canvas.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
    return canvas


def load_serif_font(font_size: int):
    from PIL import ImageFont

    names = [
        "Times New Roman Bold.ttf",
        "timesbd.ttf",
        "/usr/share/fonts/truetype/times/timesbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered_text(draw, box: tuple[int, int, int, int], text: str, font) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = x0 + max((x1 - x0 - width) // 2, 0)
    y = y0 + max((y1 - y0 - height) // 2, 0)
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


if __name__ == "__main__":
    main()
