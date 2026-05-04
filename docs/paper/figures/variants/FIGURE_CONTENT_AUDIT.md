# Figure content audit for SpikeProphecy NeurIPS submission

Follow-up to `FIGURE_GALLERY.md`. The gallery focused on *style* for
existing figures; this audit asks the harder question: **does the paper
have the right set of figures, and do they align with the paper's
strongest claims?**

Generated 2026-04-22 by Claude, overnight.

---

## Executive summary

**Diagnosis: the paper's figure set is under-aligned with its strongest
claims.**

Specifically:
1. The paper's **primary contribution** (§1, §3.3: population-metric
   decomposition) has no dedicated figure — it's introduced via
   equations and a table. Figure 1 and Figure 2 are architecture-centric,
   not protocol-centric.
2. §4.2 (cross-dataset scaling) carries the paper's sharpest "the
   decomposition reveals things aggregate r hides" argument, but it's
   **text-only** — no figure visualizes the +9% temporal / −30% spatial
   trade-off that the decomposition exposes.
3. §4.1's three-tier GLM story (Autoreg ≈ 0 / Population = −0.015 /
   Deep ≈ 0.5) is a genuinely novel finding about linear model failure
   modes, but currently lives **only in Table 1 rows and prose**.
4. The new diagonal-SSM finding (HGRN2 + Mamba + LRU cluster above
   Transformer) is **text-only in §5 Discussion** — the gallery
   proposes a new figure for this, already drafted.

**Net effect:** a reviewer skimming figures sees architecture dominance
(Figure 1) and architecture rankings (Figure 2 Pareto), but the
*decomposition protocol* — which §1 calls the primary contribution — is
not visually foregrounded anywhere. Fixing this is probably the single
biggest-impact change to the figure set.

---

## Section-by-section walk

### Abstract + §1 Introduction (no figures)

- **Primary-contribution claim:** "population metric decomposition that
  separates aggregate performance into temporal, spatial, and
  magnitude-invariant alignment."
- **Paragraph 2 (line 86)** makes a direct visual argument in words:
  "aggregate $r = 0.52$ sounds reasonable, but decomposing reveals
  $r_{pop} = 0.86$ while spatial $r = 0.54$."
- **Problem:** this example is literally described as a visual
  difference (single number vs decomposed triple) but no figure shows
  it. Reader has to hold the numbers in working memory.
- **Fix:** a small "decomposition schematic" figure next to this
  paragraph (see proposed `figure_decomp_schematic`). Costs <half a
  column; hugely improves the §1 pitch.

### §2 Related Work (no figures — correct)

No figures needed here; comparison is via Table tab:metric-compare.

### §3 Benchmark Design (no figures; has equations + tab:metric-compare, tab:architectures)

- **§3.3 Evaluation protocol (line 255)** defines the three population
  metrics (Eqs. 2–4).
- **Problem:** dense math without an accompanying visual. A reader who
  doesn't have neuroscience-imaging background will not intuit what
  `spatial_r` vs `pop_rate_r` capture geometrically.
- **Fix:** same `figure_decomp_schematic` could serve double duty —
  placed at §1 or §3.3, it shows the three axes as separate bars. If
  you want a dedicated geometric schematic (time-axis projection vs
  neuron-axis projection vs normalized projection), that would be a
  nice-to-have but bigger work.

### Figure 1 (current hero, between §3 and §4)

Currently: raster + GT/Mamba heatmaps + population-rate trace.

- **What it communicates:** "SpikeProphecy is about predicting
  population activity at scale." Decent.
- **What it doesn't communicate:** the decomposition contribution, the
  architecture comparison, any headline number beyond "scale looks big."
- Gallery already proposes 3 alternatives (minimal / story-arc /
  single-trace). Any of them is fine as a data-showcase.

**Content-audit take:** the hero figure slot is too important to spend
on a "data looks cool" figure. Consider replacing with a figure that
**carries a claim**, not just shows data. Candidates:
- The proposed `figure_decomp_schematic` (aggregate vs decomposed
  view) — directly lands the §1 primary contribution.
- A mashup: left half = data-showcase (current-ish), right half =
  decomposition schematic.

### §4.1 Architecture comparison (Table 1, no figure)

- **New finding (post-HGRN2):** "Three diagonal SSMs occupy the top
  three ranks, all above the Transformer."
- **Also new:** three-tier GLM story with clean mechanistic separation.
- **Currently figure-less** — relies on Table 1's 7 rows + prose.

**Fixes proposed:**
- Diagonal-SSM signal figure (already in gallery): `figure_diag_ssm_v1`
  / `v2` / `v3`.
- Three-tier GLM figure: proposed `figure_glm_three_tier`.
- These are complementary — they tell different parts of §4.1's story.
  I'd include **both**, not just one.

### §4.2 Cross-dataset scaling (no figure; tab:scaling in appendix)

- **Paragraph 2 (line 567–584):** makes the paper's sharpest
  "decomposition matters" argument — aggregate r barely moves
  (+4%) but decomposed metrics diverge (pop +9%, spatial −30%).
- **Currently text-only.**
- This is *the* section where the decomposition earns its keep.
  Leaving it figure-less is a missed opportunity.

**Fix:** `figure_scaling_decomp` — proposed NEW figure. Two panels:
(a) aggregate r "looks saturated"; (b) decomposed metrics show real
trade-off. This figure literally **is** the §1 contribution claim
demonstrated on real data.

### Figure 2 (current, §4.1)

Currently: Pareto + radar + bars — architecture comparison.

- Structure is good; the Pareto panel doubles as a headline chart.
- **Doesn't show HGRN2** (was created before HGRN2 existed). Needs
  regenerate.
- **Doesn't reflect diagonal/non-diagonal class coloring** — the
  gallery's new diag-SSM figures introduce that scheme; Figure 2 should
  adopt it too for cross-figure consistency.

### §4.3 Findings (Figure 3, 3 panels)

Currently: (a) region hierarchy bars, (b) Fano-stratified accuracy,
(c) population shuffle scatter.

- **Content is strong.** Three findings, each with its own panel.
- Style fixes already proposed in gallery (panel (a) dot-plot variant).
- **One nuance:** the "Finding 3: population context is essential"
  panel (c) currently shows the shuffle scatter. The three-tier GLM
  story could REPLACE or JOIN this — arguably shuffle + GLM together
  make the strongest "population context" case.

### §4.4 SNN summary → Appendix B

- Already demoted per advisor feedback.
- SNN-related figures in appendix are adequate.

### §4.5 (was: distillation) → Appendix B

- No figure. Three-row Table 6. Adequate as a negative result.

### §5 Discussion (no figures — correct for a discussion, but…)

- **Diagonal-SSM inductive-prior paragraph** is the paper's strongest
  piece of theory.
- Gallery already proposes the diag-SSM signal figure for §4.1, which
  would also serve §5 by reference.

### Appendix (figures A1–A4, tables 4–6)

- Figure A3 (AR rollout) — maybe should be promoted to main body. It's
  a striking dual-axis plot showing population predictions are robust
  while per-neuron predictions collapse. That's a substantive finding.
- Figures A1, A2, A4 appropriate as appendix material.

---

## Three NEW figures proposed (all drafted tonight)

### Figure X1: Scaling decomposition — `figure_scaling_decomp.png`

Two-panel. Left: weighted r vs session count looks flat. Right:
decomposed metrics vs session count diverge dramatically. Same data,
two views. Literally visualizes the §4.2 claim.

**Recommended placement:** §4.2, as a new Figure 3 (bumping the
current findings figure to 4).

**Why this is the single highest-impact addition:**
- It's the cleanest empirical demonstration of the paper's primary
  contribution (the decomposition).
- It's self-contained: caption alone explains the takeaway.
- Under the existing figure set, §4.2 is the only Results subsection
  that earns a real figure-worthy claim but doesn't get one.

### Figure X2: Three-tier GLM story — `figure_glm_three_tier.png`

Single panel. Three architecture classes (Autoregressive GLM,
Population GLM, Deep models). Train r vs val r as paired bars,
colored by class. Mechanism labels under each class
("no signal" / "catastrophic overfit" / "learned structure").

**Recommended placement:** §4.1, as a supplement to Table 1 rather
than a new main figure. Or move to §4.3 Finding 3 ("Population context
is essential") — it fits there thematically.

**Why worth including:** the three-tier finding is novel (we generated
the Autoreg GLM NRP job THIS WEEK; it's not in the original paper's
analysis). A figure makes the three-way comparison concrete.

### Figure X3: Decomposition schematic — `figure_decomp_schematic.png`

Two-panel. Left: a single "Weighted r = 0.52" bar — what the standard
protocol reports. Right: decomposed view showing pop_rate=0.76 /
spatial=0.55 / cosine=0.63 with interpretation labels ("temporal /
spatial / magnitude-invariant").

**Recommended placement:** §1 or §3.3. Small footprint — could be a
half-column sidebar.

**Why worth including:** the paper claims this decomposition is the
primary contribution but never shows WHY it's useful with a picture.
This figure answers "what's the difference between reporting one
number vs three?" in one glance.

---

## Current figures to rework/drop

**Rework:**
- Figure 1 — any of gallery variants, see `FIGURE_GALLERY.md`.
- Figure 2 — needs HGRN2 added + class-color scheme adopted.
- Figure 3 panel (a) — v2 dot plot from gallery.

**Consider promoting from appendix:**
- Figure A3 (AR rollout) — currently appendix; could move to main body
  as part of Findings if we have space. It's a striking plot.

**Keep as-is:**
- Appendix figures A1 (shuffle), A2 (8-metric), A4 (ceiling) — all fine.

---

## Recommended final figure set

Given 9-page NeurIPS main-body limit (E&D track: 9+refs, unlimited
appendix):

| # | Figure | Content | Source |
|---|---|---|---|
| 1 | Decomposition schematic + data showcase combo | Primary-contribution claim + data scale | `figure_decomp_schematic` combined with a mini pop-rate trace |
| 2 | Architecture comparison (Pareto) | HGRN2-updated, class-colored | rework existing Figure 2 |
| 3 | Diagonal-SSM signal | All 3 diag SSMs > non-diag | `figure_diag_ssm_v1_grouped_bars` |
| 4 | Scaling decomposition | +9%/−30% trade-off | `figure_scaling_decomp` (NEW) |
| 5 | Findings (region + Fano + shuffle OR GLM 3-tier) | Three findings | rework existing Figure 3 with dot-plot variant for panel (a) |

This is 5 main-body figures. If space is tight, drop Figure 5's panel
(c) shuffle (moved to appendix) and merge in the 3-tier GLM figure.

**Appendix additions:**
- Three-tier GLM figure (if not in main body)
- AR rollout (already there)
- Shuffle, 8-metric, ceiling (already there)

---

## Prioritization by impact

Highest-impact changes first:

1. **Add `figure_scaling_decomp`** — directly lands the primary
   contribution on the strongest argument paragraph in the Results.
   Currently text-only. **Do this first.**
2. **Replace Figure 1** with any of the gallery alternatives — first
   impressions. High visual ROI.
3. **Add `figure_diag_ssm_v1`** — lands the HGRN2 finding. Was already
   recommended in the gallery.
4. **Replace Figure 3 panel (a)** with dot plot — clean polish of an
   already-working figure.
5. **Update Figure 2** — HGRN2 + class colors for cross-figure
   consistency.
6. **Add `figure_decomp_schematic`** somewhere in §1/§3.3 — optional
   but a nice conceptual hook early in the paper.
7. **Add `figure_glm_three_tier`** — optional, strengthens §4.1.

Items 1–3 are the high-value picks. Items 4–7 are polish.

---

## Files rendered tonight (content additions)

All in `docs/neurips_ed/figures/variants/`:

- `figure_scaling_decomp.png` / `.pdf` — 2-panel aggregate-vs-decomposed scaling
- `figure_glm_three_tier.png` / `.pdf` — train/val bars per failure-mode class
- `figure_decomp_schematic.png` / `.pdf` — one-scalar vs three-axis view

Source scripts in `scripts/figures/variants/` (matching names).

All use the existing `scripts/figures/style.py` palette so they match
the rest of the paper if adopted.
