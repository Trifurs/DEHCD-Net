from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

MPL_CACHE_DIR = Path("/tmp/matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/whu/桌面/myData/Hete_CD/Haiti1")
FONT_SERIF = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]

NAVY = "#10234D"
PRE_OPTICAL_COLOR = "#10234D"
PRE_SAR_COLOR = "#2D3742"
POST_OPTICAL_COLOR = "#A94E1F"
POST_SAR_COLOR = "#315E3D"
PSEUDO_COLOR = "#B35A25"
REAL_COLOR = "#315E3D"
DIFFERENCE_COLOR = "#203A5D"
PANEL_EDGE = "#FFFFFF"
CARD_EDGE = "#D7DCE3"
PAGE_BG = "#F7F8FA"


def configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_SERIF,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a square motivation figure from Haiti1 optical/SAR remote-sensing tiles."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to the Haiti1 dataset root.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument(
        "--sample-id",
        default="1003",
        help="Tile id used for the figure. The default has clear pre-event optical imagery, cloudy post-event optical imagery, and dense change labels.",
    )
    parser.add_argument(
        "--sar-pass",
        default="desc",
        choices=("asc", "desc"),
        help="Sentinel-1 pass used for the pre/post SAR display panels.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "runs" / "figures"),
        help="Directory for exported figures.",
    )
    parser.add_argument("--name", default="haiti_motivation", help="Output filename stem.")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"], help="Formats to save.")
    parser.add_argument("--dpi", type=int, default=900, help="Raster export DPI. Values below 600 are raised to 600.")
    parser.add_argument("--size", type=float, default=7.2, help="Figure side length in inches.")
    return parser.parse_args()


def main() -> None:
    configure_matplotlib_style()
    args = parse_args()
    dpi = max(int(args.dpi), 600)
    data_root = Path(args.data_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_haiti_tile(data_root, str(args.split), str(args.sample_id), str(args.sar_pass))
    figure = draw_motivation_figure(
        arrays=arrays,
        side_inches=max(float(args.size), 1.0),
        dpi=dpi,
    )
    saved_paths: list[Path] = []
    for fmt in normalize_formats(args.formats):
        path = output_dir / f"{args.name}.{fmt}"
        save_kwargs = {"dpi": dpi, "facecolor": figure.get_facecolor(), "pad_inches": 0}
        if fmt.lower() == "png":
            save_kwargs["metadata"] = {
                "Title": "Haiti multimodal disaster motivation figure",
                "Source": f"{data_root}/{args.split}, sample {args.sample_id}",
            }
        figure.savefig(path, **save_kwargs)
        saved_paths.append(path)
    plt.close(figure)

    print("Saved Haiti motivation figure:")
    for path in saved_paths:
        print(f"  {path}")


def normalize_formats(values: Iterable[str]) -> list[str]:
    formats: list[str] = []
    for value in values:
        fmt = str(value).strip().lstrip(".").lower()
        if fmt and fmt not in formats:
            formats.append(fmt)
    return formats or ["png"]


def load_haiti_tile(data_root: Path, split: str, sample_id: str, sar_pass: str) -> dict[str, np.ndarray]:
    sar_pre_dir = "Pre_event/S1_ASC_20210805" if sar_pass == "asc" else "Pre_event/S1_DESC_20210803"
    sar_post_dir = "Post_event/S1_ASC_20210817" if sar_pass == "asc" else "Post_event/S1_DESC_20210815"
    paths = {
        "pre_optical": data_root / split / "Pre_event/S2_20210804" / f"{sample_id}.tif",
        "post_optical": data_root / split / "Post_event/S2_20210814" / f"{sample_id}.tif",
        "pre_sar": data_root / split / sar_pre_dir / f"{sample_id}.tif",
        "post_sar": data_root / split / sar_post_dir / f"{sample_id}.tif",
        "label": data_root / split / "Annotations" / f"{sample_id}.tif",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Haiti1 file(s): " + ", ".join(str(path) for path in missing))
    return {name: read_raster(path) for name, path in paths.items()}


def read_raster(path: Path) -> np.ndarray:
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    with rasterio.open(path) as src:
        return src.read()


def draw_motivation_figure(
    arrays: dict[str, np.ndarray],
    side_inches: float,
    dpi: int,
) -> plt.Figure:
    pre_optical = arrays["pre_optical"]
    post_optical = arrays["post_optical"]
    pre_sar = arrays["pre_sar"]
    post_sar = arrays["post_sar"]
    label = arrays["label"][0] if arrays["label"].ndim == 3 else arrays["label"]

    pre_cloud = pre_optical[3] != 0
    post_cloud = post_optical[3] != 0
    pre_optical_img = render_optical(pre_optical, pre_cloud)
    post_optical_img = overlay_cloud(render_optical(post_optical, post_cloud), post_cloud)
    pre_sar_img = render_sar(pre_sar)
    post_sar_img = render_sar(post_sar)

    pseudo_img = render_difference_response(pre_optical_img, pre_sar_img, tint="orange")
    real_img = render_real_change(label)
    difference_img = render_difference_response(pre_optical_img, post_sar_img, tint="blue", label=label)

    fig = plt.figure(figsize=(side_inches, side_inches), dpi=dpi, facecolor=PAGE_BG)
    draw_top_row(
        fig,
        images=[pre_optical_img, pre_sar_img, post_optical_img, post_sar_img],
    )
    draw_difference_enhancement(
        fig,
        pseudo_img=pseudo_img,
        real_img=real_img,
        difference_img=difference_img,
    )
    return fig


def draw_top_row(
    fig: plt.Figure,
    images: list[np.ndarray],
) -> None:
    titles = [
        ("Pre-optical", PRE_OPTICAL_COLOR),
        ("Pre-SAR", PRE_SAR_COLOR),
        ("Post-optical", POST_OPTICAL_COLOR),
        ("Post-SAR", POST_SAR_COLOR),
    ]
    panel_y = 0.616
    panel_w = 0.214
    gap = 0.030
    start_x = 0.029
    axes: list[plt.Axes] = []
    for idx, (image, (title, color)) in enumerate(zip(images, titles)):
        x0 = start_x + idx * (panel_w + gap)
        add_panel_shadow(fig, (x0, panel_y, panel_w, panel_w), radius=0.012)
        ax = fig.add_axes((x0, panel_y, panel_w, panel_w))
        ax.imshow(image, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        style_panel_spines(ax, linewidth=2.0)
        fig.text(
            x0 + panel_w / 2,
            panel_y + panel_w + 0.03,
            title,
            ha="center",
            va="bottom",
            fontsize=19,
            weight="bold",
            color=color,
        )
        axes.append(ax)

    add_group_bracket(fig, start_x + 0.018, start_x + 4 * panel_w + 3 * gap - 0.018, panel_y - 0.030)


def draw_difference_enhancement(
    fig: plt.Figure,
    pseudo_img: np.ndarray,
    real_img: np.ndarray,
    difference_img: np.ndarray,
) -> None:
    card = FancyBboxPatch(
        (0.035, 0.05),
        0.93,
        0.465,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor="#FFFFFF",
        edgecolor=CARD_EDGE,
        linewidth=1.3,
        zorder=-1,
        path_effects=[pe.SimplePatchShadow(offset=(1.4, -1.4), alpha=0.18), pe.Normal()],
    )
    fig.patches.append(card)

    fig.text(
        0.50,
        0.445,
        "Difference = Pseudo + Real",
        ha="center",
        va="center",
        fontsize=24,
        weight="bold",
        color=NAVY,
    )

    panel_y = 0.090
    panel_w = 0.204
    xs = [0.083, 0.398, 0.713]
    titles = [("Pseudo", PSEUDO_COLOR), ("Real", REAL_COLOR), ("Difference", DIFFERENCE_COLOR)]
    subtitles = [("modality gap", PSEUDO_COLOR), ("disaster change", REAL_COLOR), ("mixed signal", DIFFERENCE_COLOR)]
    images = [pseudo_img, real_img, difference_img]

    for x0, image, (title, color), (subtitle, subtitle_color) in zip(xs, images, titles, subtitles):
        fig.text(
            x0 + panel_w / 2,
            panel_y + panel_w + 0.040,
            title,
            ha="center",
            va="bottom",
            fontsize=20,
            weight="bold",
            color=color,
        )
        fig.text(
            x0 + panel_w / 2,
            panel_y + panel_w + 0.017,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=9.5,
            weight="bold",
            color=subtitle_color,
            alpha=0.88,
        )
        add_panel_shadow(fig, (x0, panel_y, panel_w, panel_w), radius=0.010, alpha=0.14)
        ax = fig.add_axes((x0, panel_y, panel_w, panel_w))
        ax.imshow(image, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        style_panel_spines(ax, linewidth=1.5, edgecolor="#E6E9ED")

    fig.text(0.340, 0.193, "+", ha="center", va="center", fontsize=32, weight="bold", color=NAVY)
    fig.text(0.660, 0.193, "=", ha="center", va="center", fontsize=32, weight="bold", color=NAVY)

    fig.text(0.186, 0.064, "suppress", ha="center", va="center", fontsize=9.5, weight="bold", color=PSEUDO_COLOR)
    fig.text(0.500, 0.064, "enhance", ha="center", va="center", fontsize=9.5, weight="bold", color=REAL_COLOR)


def render_optical(array: np.ndarray, invalid_mask: np.ndarray | None) -> np.ndarray:
    rgb = array[:3].astype(np.float32)
    valid = np.isfinite(rgb).all(axis=0)
    if invalid_mask is not None:
        valid &= ~invalid_mask.astype(bool)
    if int(valid.sum()) < 64:
        valid = np.isfinite(rgb).all(axis=0)

    channels: list[np.ndarray] = []
    for band in range(3):
        values = rgb[band][valid]
        if values.size:
            low, high = np.percentile(values, (1.5, 98.5))
        else:
            low, high = np.nanmin(rgb[band]), np.nanmax(rgb[band])
        channels.append(np.clip((rgb[band] - low) / (high - low + 1e-6), 0.0, 1.0))
    image = np.dstack(channels)
    gray = image.mean(axis=2, keepdims=True)
    image = np.clip(gray + (image - gray) * 1.24, 0.0, 1.0)
    return np.power(image, 0.86)


def render_sar(array: np.ndarray) -> np.ndarray:
    bands = array[:2].astype(np.float32)
    valid = np.isfinite(bands).all(axis=0)
    normalized: list[np.ndarray] = []
    for band in range(2):
        values = bands[band][valid]
        low, high = np.percentile(values, (1.5, 98.5))
        normalized.append(np.clip((bands[band] - low) / (high - low + 1e-6), 0.0, 1.0))
    vv, vh = normalized
    gray = np.power(np.clip(0.62 * vv + 0.38 * vh, 0.0, 1.0), 0.78)
    image = np.dstack([gray * 0.95, gray * 0.99, np.clip(gray * 1.08, 0.0, 1.0)])
    polarization = np.clip(vv - vh + 0.4, 0.0, 1.0)[..., None]
    warm_tint = np.dstack([np.ones_like(gray), np.ones_like(gray) * 0.95, np.ones_like(gray) * 0.88])
    return np.clip(image * (0.9 + 0.1 * polarization) + warm_tint * (0.03 * (1.0 - polarization)), 0.0, 1.0)


def overlay_cloud(image: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    mask = cloud_mask.astype(bool)
    output[mask] = output[mask] * 0.42 + np.array([1.0, 0.98, 0.92]) * 0.58
    return output


def render_difference_response(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    tint: str,
    label: np.ndarray | None = None,
) -> np.ndarray:
    reference = to_luminance(reference_image)
    target = to_luminance(target_image)
    response = normalize_map(np.abs(reference - target), lower=1.0, upper=99.0, gamma=0.72)
    if label is not None:
        real = normalize_map((label > 0).astype(np.float32), lower=0.0, upper=100.0, gamma=0.80)
        response = np.clip(0.72 * response + 0.32 * real, 0.0, 1.0)

    if tint == "orange":
        low = np.array([0.08, 0.09, 0.10])
        high = np.array([1.00, 0.66, 0.28])
    elif tint == "blue":
        low = np.array([0.08, 0.10, 0.14])
        high = np.array([0.82, 0.90, 1.00])
    else:
        low = np.array([0.0, 0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0])
    image = low + (high - low) * response[..., None]
    if label is not None:
        mask = label > 0
        image[mask] = image[mask] * 0.64 + np.array([0.56, 0.96, 0.42]) * 0.36
    return image


def render_real_change(label: np.ndarray) -> np.ndarray:
    mask = label > 0
    image = np.zeros((*label.shape, 3), dtype=np.float32)
    image[..., :] = np.array([0.025, 0.030, 0.035])
    class_colors = {
        1: np.array([0.86, 1.00, 0.62]),
        2: np.array([0.62, 0.95, 0.42]),
        3: np.array([0.96, 0.96, 0.90]),
    }
    for class_id, color in class_colors.items():
        class_mask = label == class_id
        image[class_mask] = color
    image[mask] = image[mask] * 0.90 + np.array([1.0, 1.0, 1.0]) * 0.10
    return image


def normalize_map(array: np.ndarray, lower: float, upper: float, gamma: float) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size:
        lo, hi = np.percentile(values, (lower, upper))
    else:
        lo, hi = 0.0, 1.0
    normalized = np.clip((array - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return np.power(normalized, gamma)


def to_luminance(image: np.ndarray) -> np.ndarray:
    return np.clip(0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2], 0.0, 1.0)


def add_panel_shadow(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    radius: float,
    alpha: float = 0.18,
) -> None:
    patch = FancyBboxPatch(
        (bounds[0], bounds[1]),
        bounds[2],
        bounds[3],
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor="#FFFFFF",
        edgecolor="#D9DEE5",
        linewidth=1.0,
        zorder=-2,
        path_effects=[pe.SimplePatchShadow(offset=(1.1, -1.1), alpha=alpha), pe.Normal()],
    )
    fig.patches.append(patch)


def style_panel_spines(ax: plt.Axes, linewidth: float, edgecolor: str = PANEL_EDGE) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor(edgecolor)


def add_group_bracket(fig: plt.Figure, x0: float, x1: float, y: float) -> None:
    inset = 0.008
    cap_height = 0.026
    line_y = y + 0.004
    add_figure_line(fig, [x0 + inset, x1 - inset], [line_y, line_y], linewidth=1.9)
    add_figure_line(fig, [x0 + inset, x0 + inset], [line_y, line_y + cap_height], linewidth=1.9)
    add_figure_line(fig, [x1 - inset, x1 - inset], [line_y, line_y + cap_height], linewidth=1.9)

    mid = (x0 + x1) / 2
    add_figure_line(fig, [mid, mid], [line_y - 0.002, line_y - 0.040], linewidth=1.7, alpha=0.96)
    arrow = FancyArrowPatch(
        (mid, line_y - 0.038),
        (mid, line_y - 0.066),
        transform=fig.transFigure,
        arrowstyle="Simple,tail_width=0.0,head_width=8,head_length=8",
        mutation_scale=1,
        linewidth=0,
        color=NAVY,
        zorder=2,
    )
    arrow.set_path_effects([pe.withStroke(linewidth=1.7, foreground="white", alpha=0.70)])
    fig.add_artist(arrow)


def add_figure_line(
    fig: plt.Figure,
    xs: list[float],
    ys: list[float],
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    line = Line2D(
        xs,
        ys,
        transform=fig.transFigure,
        color=NAVY,
        linewidth=linewidth,
        alpha=alpha,
        solid_capstyle="round",
        zorder=2,
    )
    line.set_path_effects([pe.withStroke(linewidth=linewidth + 1.2, foreground="white", alpha=0.72), pe.Normal()])
    fig.add_artist(line)


if __name__ == "__main__":
    main()
