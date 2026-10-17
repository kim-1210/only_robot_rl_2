from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


OUT_DIR = Path(__file__).resolve().parent


@dataclass
class BoxSpec:
    title: str
    lines: list[str]
    fill: str
    stroke: str
    width: int
    x: int = 0
    y: int = 0

    @property
    def height(self) -> int:
        return 58 + 26 * len(self.lines) + 26

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str, radius: int = 18, stroke_width: int = 2) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def text(x: int, y: int, value: str, size: int, weight: str = "600", anchor: str = "middle", fill: str = "#111827") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(value)}</text>'


def multiline(x: int, y: int, lines: list[str], size: int, gap: int, weight: str = "500", anchor: str = "middle", fill: str = "#111827") -> str:
    pieces = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">'
    ]
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else str(gap)
        pieces.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    pieces.append("</text>")
    return "".join(pieces)


def draw_box(spec: BoxSpec) -> str:
    title_y = spec.y + 34
    body_y = spec.y + 68
    return "".join(
        [
            rect(spec.x, spec.y, spec.width, spec.height, spec.fill, spec.stroke),
            text(spec.center_x, title_y, spec.title, size=17, weight="700"),
            multiline(spec.center_x, body_y, spec.lines, size=15, gap=22),
        ]
    )


def small_tag(x: int, y: int, w: int, h: int, label: str, fill: str, stroke: str) -> str:
    return rect(x, y, w, h, fill, stroke, radius=14) + text(x + w // 2, y + h // 2 + 6, label, size=15, weight="700")


def arrow(x1: int, y1: int, x2: int, y2: int, stroke: str = "#111827", width: int = 2, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" marker-end="url(#arrow)"{dash}/>'


def polyline(points: list[tuple[int, int]], stroke: str = "#111827", width: int = 2, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>'


def tag_layout(base_x: int, base_y: int, label: str, fill: str, stroke: str) -> tuple[str, int]:
    tag_y = base_y + 18
    shape = small_tag(base_x + 30, tag_y, 250, 42, label, fill, stroke)
    return shape, tag_y


def build_diagram(
    *,
    title: str,
    filename: str,
    input_box: tuple[str, list[str]],
    output_box: tuple[str, list[str]],
    cond_boxes: list[tuple[str, list[str]]],
    cond_merge: tuple[str, list[str]],
    encoder_boxes: list[tuple[str, list[str]]],
    decoder_boxes: list[tuple[str, list[str]]],
    bottleneck: tuple[str, list[str]],
    note: str,
) -> None:
    width = 1900
    height = 1180

    blue_fill = "#dbeafe"
    blue_stroke = "#2563eb"
    green_fill = "#dcfce7"
    green_stroke = "#16a34a"
    orange_fill = "#ffedd5"
    orange_stroke = "#ea580c"
    purple_fill = "#ede9fe"
    purple_stroke = "#7c3aed"
    gray_fill = "#f8fafc"
    gray_stroke = "#64748b"
    note_fill = "#fef3c7"
    note_stroke = "#b45309"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        dedent(
            """
            <defs>
              <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#ffffff"/>
                <stop offset="100%" stop-color="#f8fafc"/>
              </linearGradient>
              <marker id="arrow" markerWidth="8" markerHeight="8" refX="6.5" refY="4" orient="auto">
                <path d="M 0 0 L 8 4 L 0 8 z" fill="#111827"/>
              </marker>
            </defs>
            """
        ).strip(),
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>',
        text(width // 2, 66, title, size=42, weight="800"),
    ]

    cond1 = BoxSpec(cond_boxes[0][0], cond_boxes[0][1], purple_fill, purple_stroke, 320, 560, 110)
    cond2 = BoxSpec(cond_boxes[1][0], cond_boxes[1][1], purple_fill, purple_stroke, 420, 940, 110)
    merge = BoxSpec(cond_merge[0], cond_merge[1], "#f5f3ff", purple_stroke, 760, 570, 260)
    input_spec = BoxSpec(input_box[0], input_box[1], gray_fill, gray_stroke, 250, 30, 435)
    output_spec = BoxSpec(output_box[0], output_box[1], gray_fill, gray_stroke, 250, 1620, 435)

    enc_x = 320
    enc_w = 310
    enc_specs: list[BoxSpec] = []
    enc_y = 410
    for title_, lines in encoder_boxes:
        spec = BoxSpec(title_, lines, blue_fill, blue_stroke, enc_w, enc_x, enc_y)
        enc_specs.append(spec)
        enc_y += spec.height + 82

    dec_x = 1270
    dec_w = 330
    dec_specs: list[BoxSpec] = []
    dec_y = 410
    for title_, lines in decoder_boxes:
        spec = BoxSpec(title_, lines, green_fill, green_stroke, dec_w, dec_x, dec_y)
        dec_specs.append(spec)
        dec_y += spec.height + 82

    bottleneck_spec = BoxSpec(bottleneck[0], bottleneck[1], orange_fill, orange_stroke, 390, 755, 770)

    for spec in [cond1, cond2, merge, input_spec, output_spec, *enc_specs, *dec_specs, bottleneck_spec]:
        parts.append(draw_box(spec))

    parts.append(arrow(cond1.center_x, cond1.bottom, cond1.center_x, merge.y, stroke=purple_stroke))
    parts.append(arrow(cond2.center_x, cond2.bottom, cond2.center_x, merge.y, stroke=purple_stroke))

    parts.append(arrow(input_spec.right, input_spec.center_y, enc_specs[0].x, enc_specs[0].center_y))
    parts.append(arrow(dec_specs[0].right, dec_specs[0].center_y, output_spec.x, output_spec.center_y))

    enc_tag_fill = "#e0f2fe"
    dec_tag_fill = "#ecfccb"
    down_tags: list[tuple[int, int]] = []
    up_tags: list[tuple[int, int]] = []

    for idx, spec in enumerate(enc_specs[:-1]):
        tag_svg, tag_y = tag_layout(enc_x, spec.bottom, f"Downsample {idx + 1}", enc_tag_fill, blue_stroke)
        down_tags.append((enc_x + 155, tag_y))
        parts.append(tag_svg)
        parts.append(arrow(spec.center_x, spec.bottom, spec.center_x, tag_y, stroke="#1d4ed8"))
        parts.append(arrow(spec.center_x, tag_y + 42, spec.center_x, enc_specs[idx + 1].y, stroke="#1d4ed8"))

    for idx, spec in enumerate(dec_specs[:-1]):
        tag_svg, tag_y = tag_layout(dec_x + 20, spec.bottom, f"Upsample {len(dec_specs) - idx - 1}", dec_tag_fill, green_stroke)
        up_tags.append((dec_x + 175, tag_y))
        parts.append(tag_svg)
        parts.append(arrow(spec.center_x, spec.bottom, spec.center_x, tag_y, stroke="#15803d"))
        parts.append(arrow(spec.center_x, tag_y + 42, spec.center_x, dec_specs[idx + 1].y, stroke="#15803d"))

    parts.append(arrow(enc_specs[-1].right, enc_specs[-1].center_y, bottleneck_spec.x, bottleneck_spec.center_y))
    parts.append(arrow(bottleneck_spec.right, bottleneck_spec.center_y, dec_specs[-1].x, dec_specs[-1].center_y))

    skip_trunk_x = 960
    parts.append(polyline([(enc_specs[0].right, enc_specs[0].center_y), (skip_trunk_x, enc_specs[0].center_y), (skip_trunk_x, dec_specs[0].center_y), (dec_specs[0].x, dec_specs[0].center_y)]))
    parts.append(polyline([(enc_specs[1].right, enc_specs[1].center_y), (skip_trunk_x - 40, enc_specs[1].center_y), (skip_trunk_x - 40, dec_specs[1].center_y), (dec_specs[1].x, dec_specs[1].center_y)]))
    parts.append(polyline([(enc_specs[2].right, enc_specs[2].center_y), (skip_trunk_x - 80, enc_specs[2].center_y), (skip_trunk_x - 80, dec_specs[2].center_y), (dec_specs[2].x, dec_specs[2].center_y)]))

    cond_trunk_y = merge.bottom + 24
    cond_columns = [500, 690, 900, 1110, 1380, 1490, 1600]
    cond_targets = [
        (enc_specs[0].center_x, enc_specs[0].y),
        (enc_specs[1].center_x, enc_specs[1].y),
        (enc_specs[2].center_x, enc_specs[2].y),
        (bottleneck_spec.center_x, bottleneck_spec.y),
        (dec_specs[0].center_x, dec_specs[0].y),
        (dec_specs[1].center_x, dec_specs[1].y),
        (dec_specs[2].center_x, dec_specs[2].y),
    ]
    for trunk_x, (tx, ty) in zip(cond_columns, cond_targets):
        parts.append(polyline([(merge.center_x, cond_trunk_y), (trunk_x, cond_trunk_y), (trunk_x, ty - 24), (tx, ty - 24), (tx, ty)], stroke=purple_stroke, width=2, dashed=True))

    parts.append(rect(430, 1060, 1040, 76, note_fill, note_stroke, radius=20))
    parts.append(text(width // 2, 1108, note, size=24, weight="700"))
    parts.append("</svg>")

    (OUT_DIR / filename).write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    build_diagram(
        title="Reach Task - Diffusion Policy U-Net",
        filename="reach_diffusion_architecture.svg",
        input_box=("Noisy action chunk x_t", ["shape [B, 7, 16]"]),
        output_box=("Predicted noise epsilon", ["shape [B, 7, 16]"]),
        cond_boxes=[
            ("Diffusion step t", ["sinusoidal embedding", "time emb dim 256"]),
            ("Obs encoder", ["28D obs = joint pos 7 + joint vel 7", "+ pose cmd 7 + prev joints 7"]),
        ],
        cond_merge=("Global condition", ["time emb 256 + obs emb 28"]),
        encoder_boxes=[
            ("Encoder stage 1", ["2x Conditional ResBlock1D", "7 -> 128"]),
            ("Encoder stage 2", ["2x Conditional ResBlock1D", "128 -> 256"]),
            ("Encoder stage 3", ["2x Conditional ResBlock1D", "256 -> 512"]),
        ],
        decoder_boxes=[
            ("Decoder stage 1", ["concat skip +", "2x Conditional ResBlock1D", "512 + 512 -> 256"]),
            ("Decoder stage 2", ["concat skip +", "2x Conditional ResBlock1D", "256 + 256 -> 128"]),
            ("Output refinement", ["Conv1dBlock + 1x1 Conv", "128 -> 7"]),
        ],
        bottleneck=("Bottleneck", ["2x Conditional ResBlock1D", "512 channels"]),
        note="FiLM conditioning is injected into every residual block",
    )

    build_diagram(
        title="Reach Task - Flow Matching U-Net",
        filename="reach_flow_matching_architecture.svg",
        input_box=("Flow state x_t", ["shape [B, 7, 16]", "noise-action interpolation"]),
        output_box=("Predicted velocity v_theta", ["shape [B, 7, 16]"]),
        cond_boxes=[
            ("Flow time t", ["sinusoidal embedding", "time emb dim 128"]),
            ("Obs encoder", ["28D obs = joint pos 7 + joint vel 7", "+ pose cmd 7 + prev joints 7"]),
        ],
        cond_merge=("Condition MLP", ["time emb 128 + obs emb 128", "projected to base channels 128"]),
        encoder_boxes=[
            ("Encoder stage 1", ["input proj + ResBlock1D", "7 -> 128"]),
            ("Encoder stage 2", ["ResBlock1D", "128 -> 256"]),
            ("Bottleneck input", ["downsampled latent", "256 channels"]),
        ],
        decoder_boxes=[
            ("Decoder stage 1", ["upsample + concat skip +", "ResBlock1D", "256 + 256 -> 128"]),
            ("Decoder stage 2", ["upsample + concat skip +", "ResBlock1D", "128 + 128 -> 128"]),
            ("Output head", ["GroupNorm + SiLU + Conv1d", "128 -> 7"]),
        ],
        bottleneck=("Mid blocks", ["2x ResBlock1D", "256 channels"]),
        note="Condition vector is added inside every ResBlock1D via cond_proj",
    )

    build_diagram(
        title="Reach-Lift Task - Diffusion Policy U-Net",
        filename="reach_lift_diffusion_architecture.svg",
        input_box=("Noisy action chunk x_t", ["shape [B, 7, 8]"]),
        output_box=("Predicted noise epsilon", ["shape [B, 7, 8]"]),
        cond_boxes=[
            ("Diffusion step t", ["sinusoidal embedding", "time emb dim 384"]),
            ("Obs history encoder", ["124D obs = 4 x 31D frames", "frame = joint pos 7 + joint vel 7", "+ object pos 3 + pose cmd 7 + prev joints 7"]),
        ],
        cond_merge=("Global condition", ["time emb 384 + obs emb 124", "weighted around grasp/lift windows"]),
        encoder_boxes=[
            ("Encoder stage 1", ["2x Conditional ResBlock1D", "7 -> 192"]),
            ("Encoder stage 2", ["2x Conditional ResBlock1D", "192 -> 384"]),
            ("Encoder stage 3", ["2x Conditional ResBlock1D", "384 -> 768"]),
        ],
        decoder_boxes=[
            ("Decoder stage 1", ["concat skip +", "2x Conditional ResBlock1D", "768 + 768 -> 384"]),
            ("Decoder stage 2", ["concat skip +", "2x Conditional ResBlock1D", "384 + 384 -> 192"]),
            ("Output refinement", ["Conv1dBlock + 1x1 Conv", "192 -> 7"]),
        ],
        bottleneck=("Bottleneck", ["2x Conditional ResBlock1D", "768 channels"]),
        note="Reach-lift adds 4-step observation history and task-aware training weights",
    )

    build_diagram(
        title="Reach-Lift Task - Flow Matching U-Net",
        filename="reach_lift_flow_matching_architecture.svg",
        input_box=("Flow state x_t", ["shape [B, 8, 8]", "zero/noise initialized trajectory"]),
        output_box=("Predicted velocity v_theta", ["shape [B, 8, 8]"]),
        cond_boxes=[
            ("Flow time t", ["sinusoidal embedding", "time emb dim 256"]),
            ("Obs history encoder", ["136D obs = 4 x 34D frames", "frame = joint pos 8 + joint vel 8", "+ object pos 3 + pose cmd 7 + prev joints 8"]),
        ],
        cond_merge=("Condition MLP", ["time emb 256 + obs emb 256", "projected to base channels 256"]),
        encoder_boxes=[
            ("Encoder stage 1", ["input proj + ResBlock1D", "8 -> 256"]),
            ("Encoder stage 2", ["ResBlock1D", "256 -> 512"]),
            ("Bottleneck input", ["downsampled latent", "512 channels"]),
        ],
        decoder_boxes=[
            ("Decoder stage 1", ["upsample + concat skip +", "ResBlock1D", "512 + 512 -> 256"]),
            ("Decoder stage 2", ["upsample + concat skip +", "ResBlock1D", "256 + 256 -> 256"]),
            ("Output head", ["GroupNorm + SiLU + Conv1d", "256 -> 8"]),
        ],
        bottleneck=("Mid blocks", ["2x ResBlock1D", "512 channels"]),
        note="Two gripper channels are preserved and weighted more heavily in training",
    )


if __name__ == "__main__":
    main()
