# Claude Design prompt — SpikeProphecy Figure 1 graphical abstract

**Paste the block below into Claude Design.** It specifies layout, text,
colors, and data values. Iterate by adjusting individual sections.

---

## Prompt

Design a scientific graphical abstract for a NeurIPS 2026
Evaluations-and-Datasets-Track paper titled **"SpikeProphecy: A
Large-Scale Benchmark for Neural Population Forecasting"**.

Aesthetic target: the POYO paper's Figure 1 style (Nature-methods
graphical abstract). Clean horizontal flow diagram. Real data
visualizations embedded inside each region (not just icons). Thin clean
borders, white background, sans-serif typography. Mathematical notation
where appropriate. Minimal decorative ornament. Feels like a published
Nature figure, not an infographic.

### Canvas

- Aspect ratio: approximately **2:1 landscape** (e.g. 2400 × 1200 px at
  300 DPI, final print ~6.5 in × 3.25 in for a NeurIPS single-column
  figure).
- Background: pure white (#FFFFFF).
- Safe margins: 4% padding on all sides.

### Layout

Three horizontally arranged regions with arrows connecting them:

```
┌──────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│   STAGE 1        │ ───▶ │     STAGE 2          │ ───▶ │    STAGE 3       │
│   Benchmark      │      │ Decomposition        │      │    Findings      │
│   (blue)         │      │ protocol  (orange,   │      │    (green)       │
│                  │      │ visually emphasized) │      │                  │
└──────────────────┘      └──────────────────────┘      └──────────────────┘
```

Region widths (of the drawable canvas width):
- Stage 1: 28%
- Gap + arrow: 4%
- Stage 2: 36% (widest — it is the highlight)
- Gap + arrow: 4%
- Stage 3: 28%

Each stage is a rounded rectangle (corner radius ~12 px). Stage 2 has a
slightly thicker border (2.5 px) than stages 1 and 3 (1.5 px) to
emphasize that it is the paper's central contribution.

### Overall title

Above all three stages, centered:

> **SpikeProphecy | decomposition at the center of the benchmark**

Font: sans-serif (Helvetica / Inter / Arial), weight bold, size ~16 px,
color `#222222`. Do not add a subtitle.

---

### STAGE 1 — "Benchmark inputs" (blue)

- Background fill: `#EAF3FA`
- Border color: `#2E6FA7`
- Stage title at top-left inside the box: "**1. Benchmark**"
  (bold, `#2E6FA7`, ~11 px)

Three elements stacked vertically inside:

**(a) Mini Neuropixels raster image** (top element, ~55% of stage height):
- A rectangular heatmap representing ~100 neurons × 200 time bins of
  spike-count data.
- Colormap: dark purple/black at zero, transitioning through magenta to
  warm orange/yellow at high values. Think "spike activity" feel. Exact
  colormap stops (low → high):
  `#0d0221 → #2a0845 → #6b1d5e → #c94277 → #f4a236 → #ffd166`.
- White-text overlay in the raster's upper-left corner (small, ~7 px):
  "Neuropixels | 50 ms bins"
- A thin dark-blue border around the raster (~1 px).

**(b) Dataset summary line** (middle element, single line of text):
> "Steinmetz + IBL  |  105 sessions  |  ~89,800 neurons"

Font: sans-serif regular, ~8 px, `#233A52`, centered.

**(c) 7-architecture colored chip strip** (bottom element):
- Seven rounded-rectangle chips in a horizontal row, equal width, small
  gap between them.
- Chip height: ~20 px. Chip text: white, bold, sans-serif, ~7 px,
  centered.
- Order and colors (left to right):

  | Label  | Fill color | Architecture family |
  |--------|-----------|---------------------|
  | Mamba  | `#0072B2` | Diagonal SSM |
  | HGRN2  | `#0072B2` | Diagonal SSM |
  | Tfmr   | `#D55E00` | Attention |
  | GDelta | `#882255` | Non-diag SSM |
  | LRU    | `#0072B2` | Diagonal SSM |
  | LSTM   | `#E69F00` | Gated RNN |
  | SNN    | `#009E73` | Spiking |

- Italic caption below the chip strip (~7 px, `#233A52`):
> "7 baselines  |  4 structural families  |  matched 3-layer training"

---

### STAGE 2 — "Decomposition protocol" (orange, HIGHLIGHTED)

- Background fill: `#FFF3E0`
- Border color: `#D55E00` (2.5 px — visually emphasized)
- Stage title at top-left inside the box: "**2. Decomposition protocol**"
  (bold, `#D55E00`, ~11 px)
- Immediately below title in slightly smaller italic: "*our contribution*"
  (`#D55E00`, ~8 px)

Two elements stacked vertically inside:

**(a) Central formula** (top half):

Centered equation in large typeface (~14 px):

> aggregate **r**   ⟶   ( *r*<sub>pop</sub>,  *r*<sub>spatial</sub>,  cos-sim )

Render with italic math style for the r variables and subscripts. Arrow
is a long right arrow (→).

Under the formula, italic caption (~8 px, `#D55E00`, centered):
> "three orthogonal axes of population fidelity"

**(b) Three mini visualizations** (bottom half) — three equally sized
sub-panels in a row, each with a bold orange label (`#D55E00`) above:

1. **Label: "*r*<sub>pop</sub>"**
   Mini timeseries line plot. Gray semi-transparent line (GT population
   rate) with a bold orange line on top (model prediction). Lines closely
   overlap to imply good fit. Light fill under gray GT line at low alpha.
   No axis ticks — just a clean rectangle with the traces inside.

2. **Label: "*r*<sub>spatial</sub>"**
   Mini scatter plot. ~80 orange dots at low alpha, clustered near the
   y = x diagonal (which is a thin dashed gray line). Dots show some
   scatter, roughly from (0,0) to (1,1) corner.

3. **Label: "cos-sim"**
   Two vectors originating from (0,0):
     - Dark gray vector pointing up-and-right at ~65° (length 1).
     - Bold orange vector pointing up-and-right at ~45° (length 1).
   Small arc connecting them with Greek letter θ labeled inside the arc.
   Clean rectangle frame, no ticks.

---

### STAGE 3 — "Findings" (green)

- Background fill: `#EAF5EE`
- Border color: `#2E8B57` (1.5 px)
- Stage title at top-left inside the box: "**3. Findings**"
  (bold, `#2E8B57`, ~11 px)

Four stacked findings vertically. Each finding is one row with three
elements left-to-right: bullet dot, text block, mini chart.

**Format per row:**
- Green bullet circle (`#2E8B57`, ~6 px diameter) on left
- Finding title (bold, `#2E8B57`, ~9 px)
- Finding detail line below title (regular, `#2C4A3A`, ~7 px)
- Mini chart on the right side of the row

**Row 1 — Brain-region hierarchy:**
- Title: "Brain-region hierarchy"
- Detail: "ANCOVA R² = 0.28,  p < 10⁻⁷⁷  (8 functional systems)"
- Mini chart: 8-bar horizontal or vertical bar chart. Top 3 bars in
  saturated green (`#2E8B57`), bottom 5 bars in pale muted green
  (`#B0C4B1`). Bars descend in height left to right. Bar values
  (top to bottom): 0.158, 0.155, 0.139, 0.131, 0.129, 0.122, 0.104, 0.104.

**Row 2 — Modern-recurrence tier:**
- Title: "Modern-recurrence tier"
- Detail: "5 architectures within 2 pp;  Mamba leads by < 1 pp"
- Mini chart: 7 vertical bars colored by family (same palette as Stage 1
  chips). Heights (left to right, Mamba→SNN): 0.500, 0.493, 0.492, 0.485,
  0.480, 0.441, 0.430. Y-axis range implied 0.40–0.52.

**Row 3 — Sub-Poisson evaluation floor:**
- Title: "Sub-Poisson floor"
- Detail: "28% of neurons (FF < 1);  mean r = 0.07"
- Mini chart: Fano-factor histogram. Light-gray bars with a dashed
  vertical line at Fano factor = 1.0 in saturated green. Distribution is
  bimodal, with ~28% of mass left of the line.

**Row 4 — KD null result:**
- Title: "KD null result"
- Detail: "standalone ≥ all distilled variants in count regression"
- Mini chart: 3 bars. First bar (Standalone) in saturated green
  (`#2E8B57`) at height 0.477. Second bar (KD β=0.5) in pale green
  (`#B0C4B1`) at height 0.475. Third bar (KD β=1.0) in pale green at
  height 0.458. Label under bars: "β=0 | 0.5 | 1.0".

---

### Arrows (between stages)

Two rightward arrows connect the three stages horizontally, at mid-height
of the boxes.

- Arrow 1 (Stage 1 → Stage 2): blue (`#2E6FA7`), ~2 px thick, arrowhead
  length ~12 px. Small italic label centered above the arrow:
  *"evaluated with"* (~7 px, blue).
- Arrow 2 (Stage 2 → Stage 3): orange (`#D55E00`), ~2 px thick. Label
  above: *"yields"* (~7 px, orange).

Arrows should terminate cleanly at the box edges, not inside or outside.

---

### Typography

All text in a **sans-serif** family (Helvetica, Inter, Arial). Weights:

- Main title: bold
- Stage headers: bold
- Finding titles: bold
- Formula variables: italic (math style)
- Captions / subtitles: regular italic
- Chip labels: bold

No mixed font families. No decorative display fonts.

### Style notes

- Do **not** add drop shadows, gradients, glow effects, or 3D
  perspective.
- Do **not** use stock icons (people, buildings, lightbulbs). Only use
  actual data visualizations or mathematical notation as visuals.
- Do **not** use clip art or cartoon brain illustrations.
- Line weights: 1–1.5 px for borders, 1–2 px for arrows, 0.5–1 px for
  data-viz strokes.
- Keep the entire figure feeling "data-dense but restrained" — the
  POYO aesthetic.

---

## Iteration notes for the designer

- If Stage 2 looks too text-heavy, move the caption to a single line
  under the formula.
- The chip strip in Stage 1 is the only place architecture names appear;
  they're not repeated elsewhere.
- Colors are deliberate, not decorative. The **blue = input**,
  **orange = contribution**, **green = finding** mapping should be
  immediately readable.
- If the three mini visualizations in Stage 2 feel crammed, reduce their
  combined width to ~70% of the stage and keep the formula prominent.
