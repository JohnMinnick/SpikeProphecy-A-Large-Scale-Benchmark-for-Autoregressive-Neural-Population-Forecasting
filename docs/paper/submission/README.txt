SpikeProphecy -- NeurIPS 2026 E&D Track submission materials
============================================================

Plain-text copies of the metadata fields you'll paste into the
OpenReview submission form. The PDF (../main.pdf) is the actual
submission artifact; everything in this folder is the wrapper.

Files in this folder
--------------------

  title.txt       Single-line paper title (matches main.tex \title{}).
  abstract.txt    Abstract with LaTeX inline math preserved
                  ($r$, ${\sim}89{,}800$, $\Delta R^{2}{=}0.018$,
                  $\to$, etc.). OpenReview renders LaTeX in the
                  abstract field, so paste this verbatim.
  authors.txt     Final author order, affiliation, and a per-author
                  email/ORCID slot to fill in as coauthors confirm.
  keywords.txt    Free-text keyword list + suggested primary &
                  secondary subject areas.
  tldr.txt        Short summary (~250 char) for the OpenReview
                  TL;DR field.

Pre-submission checklist
------------------------

  [ ] Confirm author emails + ORCIDs in authors.txt
  [ ] Confirm primary subject area with E&D track CFP page
  [ ] Verify main.tex is built with [main] (anonymous), not
      [eandd, preprint], for the actual upload
  [ ] Re-run leakage audit suite (pytest tests/test_leakage*.py)
      and confirm pass state
  [ ] Source-data links verified live (Figshare DOI for Steinmetz,
      IBL ONE API for IBL Repeated Site)
  [ ] pip-installable evaluation toolkit (`pip install spikeprophecy`)
      is published
  [ ] GitHub repo anonymized for review (separate `submission`
      branch with author-identifying material stripped)

Track policies
--------------

NeurIPS 2026 E&D Track is double-blind for review. Submissions must:
  - Use the [eandd] template option (not [preprint])
  - Provide public links to the source datasets used (no
    re-hosting required; we only redistribute our processed tensors
    and preprocessing code)
  - Pass the NeurIPS Paper Checklist (see ../checklist.tex)

Sources of truth
----------------

The plain-text files here are derived from main.tex; if main.tex
changes (title, author order, abstract), regenerate these files
to match. Do NOT hand-edit them out of sync with the LaTeX source.
