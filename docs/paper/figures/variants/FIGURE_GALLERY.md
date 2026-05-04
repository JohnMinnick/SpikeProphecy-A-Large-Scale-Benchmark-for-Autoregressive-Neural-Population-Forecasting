# Figure design gallery for SpikeProphecy NeurIPS submission

Generated 2026-04-22 overnight by Claude. Eight figure variants across
three figure slots, plus my read on each. **You pick winners; I apply
to the paper.**

All variants are real-data renders (Steinmetz session 4 from S3 cache
for Figure 1; Table 1 / pop_metrics JSON values for the rest).

---

## NEW figure: Diagonal-SSM signal (3 variants)

This figure does not exist in the current paper — it visualizes the
new diagonal-SSM finding from the HGRN2 run. **My strong vote: include
one of these in the main body** (currently the diagonal-SSM finding
lives only in Discussion prose, which reviewers can miss).

Recommended placement: alongside or just after Table 1 in §4.1.

### Variant 1 — Grouped bars *(my pick)*

`figure_diag_ssm_v1_grouped_bars.png`

Single-panel bar chart, 6 architectures grouped by structural class
(Diagonal SSM / Attention / Gated RNN / Spiking). Horizontal band
shades the diagonal-SSM cluster. Mamba's r is a dotted reference line.

**Pros:**
- Strongest 3-second takeaway: viewer instantly sees "blue bars on top, others below."
- Color + hatching redundancy works for colorblind readers.
- Numbers labeled on each bar — no squinting at axis.
- The shaded band literally draws the cluster.

**Cons:**
- Loses the parameter-efficiency story (no x-axis for params).
- The "Diagonal SSMs" call-out arrow is slightly cluttered — could remove.

### Variant 2 — Colored Pareto

`figure_diag_ssm_v2_pareto.png`

2D scatter: weighted r vs parameters. Shaded "diagonal-SSM frontier"
region. Each model labeled in-place with class color + shape.

**Pros:**
- Combines efficiency + performance story in one panel.
- Diagonal SSMs visibly cluster in upper-right region.

**Cons:**
- Frontier polygon is slightly visually misleading — it shades the
  whole upper-right region, not just where diagonal SSMs are. Could be
  fixed by changing the polygon to hug only the points.
- Less direct than the bars for the headline message.

### Variant 3 — Multi-metric dot plot

`figure_diag_ssm_v3_multimetric.png`

Four small panels (one per metric), each a horizontal dot plot of all
6 models. Background band over diagonal-SSM rows shows they cluster
across all four metrics — strongest version of "HGRN2 ≈ Mamba."

**Pros:**
- Most rigorous: shows the diagonal cluster holds across *every*
  decomposed metric, not just weighted r.
- Reinforces the protocol claim (decomposition matters) by using it.

**Cons:**
- Connecting line between dots could imply ordering relationship
  between models (it doesn't — they're discrete). Easy to remove.
- Less immediate visual punch than v1.

---

## Figure 1 hero (3 variants)

Current figure: `../figure1_benchmark_overview.png`.

The current figure has a 3-panel layout (raster + heatmaps + pop trace)
that I find visually busy. The variants below explore alternatives.

### Variant 1 — Minimal 2-panel Nature style *(my pick if you keep heatmaps)*

`figure1_hero_v1_minimal.png`

Top panel: clean population rate trace with GT + Mamba + SNN, legend
outside-right, scale annotation in upper-left. Bottom panel: GT vs
Mamba prediction heatmaps side-by-side.

**Pros:**
- Big, clean pop trace as the lead — communicates the headline result
  (r_pop = 0.98) instantly.
- Drops the standalone raster (was redundant with heatmap).
- Nature-style: simple, deliberate, reads top-to-bottom.

**Cons:**
- Heatmap pair takes more vertical space than warranted given the dense
  appearance — most of the viewer's attention goes to the bright
  vertical bursts, not the spike-by-spike comparison.
- Doesn't show HGRN2 (would need 4 lines on the trace; current shows GT,
  Mamba, SNN).

### Variant 2 — 3-panel story arc *(closest to current, polished)*

`figure1_hero_v2_storyarc.png`

(a) Zoomed raster, (b) GT vs Mamba heatmaps, (c) decomposed-metric
bar triptych for all 6 architectures. Tells: data → prediction → eval.

**Pros:**
- Most "complete" figure — shows the input data, the prediction, and
  the quantitative result all in one view.
- Panel (c) brings the architecture comparison into the hero figure
  rather than burying it in Table 1 — high reviewer impact.
- Shows HGRN2 in the bar charts.

**Cons:**
- Densest of the three; some reviewers will skim past it.
- Panel (a) raster is uniformly orange (high-rate neurons dominate the
  color scale) — less informative than I hoped.

### Variant 3 — Single-panel striking trace *(my pick if you want maximum first impression)*

`figure1_hero_v3_singletrace.png`

ONE panel: the population trace, GT filled + Mamba + SNN overlaid.
No heatmap, no raster. Caption carries scale info.

**Pros:**
- **Maximum visual impact** — fills the page width with the trace,
  the headline number (r_pop = 0.98) is impossible to miss.
- Forces the eye to the actual claim: "we predict population dynamics
  faithfully at scale."
- Minimal cognitive load — reviewer doesn't have to figure out which
  panel matters.

**Cons:**
- Trades visual breadth for impact. Loses the per-neuron heatmap
  visualization entirely.
- Some reviewers will want to see "what does the data look like?" and
  this doesn't directly show that — relies on caption text.

---

## Figure 3 panel (a) — Brain region hierarchy (2 variants)

Current panel (a) had cramped rotated x-axis labels (we partially fixed
by shortening to abbreviations like "Sens.", "Motor", etc. but the
fundamental geometry is still tight). Variants below switch to
horizontal layout.

### Variant 1 — Horizontal bars

`figure3_panela_v1_horizontal.png`

Each region is a row. Raw and ANCOVA-adjusted shown as paired bars,
with values labeled at the end.

**Pros:**
- Region names readable at full length (no rotation).
- Numbers labeled — viewer doesn't need axis.

**Cons:**
- Bars feel slightly heavy compared to a dot plot.
- The "ANCOVA shift" is implicit (bar lengths) rather than explicit.

### Variant 2 — Dot plot with shift lines *(my pick)*

`figure3_panela_v2_dotplot.png`

Each region row has Raw (open orange) and Adjusted (filled blue) dots
connected by a gray line showing the ANCOVA correction direction and
magnitude. Sorted by adjusted r.

**Pros:**
- Cleanest. Shift direction (left/right) and magnitude (line length)
  are visually explicit — exactly the story §4.3 wants to tell.
- Sort order = ranking by the substantive statistic (adjusted r).
- Minimal ink, NeurIPS production quality.

**Cons:**
- Reader needs to learn the convention (dots + lines) — slightly less
  immediate than bars for someone who skims fast.

---

## My overall recommendation

If you want to keep the figure changes minimal but high-impact:

1. **Add the new diagonal-SSM signal figure (v1 grouped bars) as
   Figure 4.** Place after the architecture-comparison results in §4.1.
   Bumps the diagonal-SSM finding from a Discussion claim to a visual
   front-and-center result.

2. **Replace Figure 1 with Variant 1 (minimal 2-panel) OR Variant 3
   (single trace).** Variant 1 if you want to keep some "data
   visualization" feel; Variant 3 if you want maximum reviewer-first-
   impression impact.

3. **Replace Figure 3 panel (a) with the v2 dot plot.** Subtle but
   clear improvement; eliminates the cramped-label issue.

Total: one new figure, two replacements. ~30 minutes to wire into
`main.tex` once you pick.

**If you want to be more aggressive:**
- Use Figure 1 v2 (story arc) — packs the most info but is densest.
  Good if you're confident reviewers will engage carefully.
- Use diag-SSM v3 (multi-metric) instead of v1 — stronger rigor signal,
  reinforces the decomposition protocol.

**Smaller things I would also do** (not in scope tonight, but worth
flagging):
- Switch from sans-serif to a serif or stix-mathfont in figures, so
  figures match the paper body. Low priority.
- Add tiny error bars / confidence bands once you do the seed sweep
  recommended in Limitations.
- The current Figure 2 (Pareto + radar + bars) doesn't show HGRN2.
  Could re-color those panels to use the diagonal/non-diagonal class
  scheme from the new variants — would visually unify the
  architectural narrative across figures.
