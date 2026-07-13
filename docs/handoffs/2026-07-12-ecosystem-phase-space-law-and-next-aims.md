# Ecosystem handoff — 2026-07-12 — the phase-space law, and what to run next

**Scope:** cross-repo capstone for the evening (sk-mcp · CDG · the pools · Cathedral).
The **repo-local** sk-mcp handoff (`docs/handoffs/2026-07-12-cone-null-...`) covers the
sk-mcp-specific work; **this** ties the repos together and sets next session's aims.
*(Homed in sk-mcp `docs/handoffs/` as the ecosystem-scoped companion to the repo-local handoff.)*

---

## The law that emerged (read this first)

> **Register is a stratified phase space. Different axes freeze at different times, and you can
> only read it contrastively against a learned null.**

It landed in three repos independently the same night, which is why it's a *law* and not a pun:
- **sk-mcp / co-adaptation** — an operator's register saturating into their *mean* over years
  (freeze = ritualization; "Good!" → μ → turn-signal-at-3am).
- **CDG / liquid-state decoding** — a token sublimes frozen↔steam with no liquid basin; and the lit
  (`Steering Without Breaking`, 2605.10971) shows attributes commit at different schedule points
  (topic <2%, sentiment ~20%) — *each axis has its own freeze-time*, at the scale of one step.
- **Methodological floor, identical** — the null **is** the experiment; absolute measurement
  reproduces the confound (why ADR-SKMCP-0005's absolute affect died, contrastive survived). CDG's
  H0s are all differentials too. Both repos rediscovered *read the distribution, not the scalar*.

**The loop it closes:** sk-mcp is the **sensor** (detect entrainment), CDG's liquid-phase is the
inverse **actuator** (give register *choice* back instead of a premature single commit), the
Cathedral's State-Based Resonance is the **governance** — *iff cone-nulled, or it becomes the echo
chamber its own spec fears (R06).*

---

## NEXT SESSION — aims, ranked

### AIM 1 — Run the idea-corpus experiments. **(Start here: zero setup, everything staged.)**
The store is embedded and the H0s are locked. Just run them.
- **Substrate:** `design-docs/experiments/idea-corpus-nv4096/vectors.jsonl` (419 nv-embed chunks,
  `model_name = NVEmbed:nvidia/NV-Embed-v2` — **the nv4096 self-null applies**). `measure_cone.py`
  smoke-passes on it.
- **Locked pre-registration:** `design-docs/experiments/idea-corpus-latent-axes/PRE-REGISTRATION.md`
  (committed `b1964e8` *before* the store existed — the pawl).
- **Do:** EXP-A latent axes (`measure_cone.py` + `probe_axis_poles.py --corpus .../idea-corpus-nv4096/vectors.jsonl`;
  H0: PC1 = genre/envelope, not topic — the claim rides the **neighbor-triple contrast**, not the
  spectrum). EXP-B tacit-link recovery (start with the `provenance-context-handling` symlink specimen +
  3–4 named links; semantic recall vs keyword recall). EXP-C void-as-triage (does confidence separate
  `user:gate` from `auto:fix` above noise; triage labels are the *independent* ground truth — hold out,
  don't fit).

### AIM 2 — Pre-register the cross-scale isomorphism **with its brake, before running.**
Does the human register **freeze-ordering** match CDG's diffusion **commit-ordering** (topic-before-
sentiment)? A *law* only if orderings match **by a shared mechanism** (early commits constrain late
ones); otherwise an analogy. **Write the does-not-hold condition first** (fails if orderings differ,
OR match by unshared mechanisms). Extends co-adaptation Node A from *one* axis to a **per-axis
freeze-time spectrum**. New cross-repo record in `experiments/`.

### AIM 3 — Co-adaptation Node A first cut (reuse-first).
Recover the entrainment prior-art in `design-docs/incoming-ideas/images/`
(`exclamation_entrainment_with_releases.png`, `who_leads_analysis.png`, `chart1_inverse_correlation.png`)
and the generating script — **it may also be Node B's lost variability encoder**
(`semantic-chunker/scripts/forensics/structural_fingerprint.py` is the lead). Then the
release-aligned lag/overshoot on the timestamped `exclamation_analysis_2023-2025.json`.

### AIM 4 — (CDG repo, gated) build the `DISTRIBUTION` socket.
CDG's liquid H0-observe/project are **blocked** until per-position top-k+weights are logged
(current logging is committed-state-only — "the liquid is invisible except where it leaks as churn").
That's CDG-repo work; see `ComfyUI-DiffusionGemma/docs/experiments/` (the liquid note + its
`experiment.md`, 5 H0s). Pointer only — not seat work.

---

## What's durable now (grounded, with SHAs)

- **sk-mcp** — `measure_cone.py`, `probe_axis_poles.py`, `docs/research/co-adaptation-longitudinal.md`,
  `docs/runbooks/anisotropy-instruments.md`, `docs/map.md` (cone-block corrected 80×→160×), the
  repo handoff, and **this file** — on branch **`session/2026-07-12-anisotropy-instruments`**
  (`d1170f7` scripts, `d4b950e` docs, + this handoff), **pushed**; `main` deliberately untouched.
- **design-docs** — idea-corpus **pre-registration** committed `b1964e8` on `main`, pushed, registered
  in `experiments/README.md`. Embedded store `idea-corpus-nv4096/` present (untracked substrate,
  regenerable in ~51s).
- **CDG** — operator-authored liquid-state note + `experiment.md` (5 H0s) live in-repo.

## Durable-capture debts (live only in synthesis right now — bank when fresh)

- **The phase-space law** ↑ — needs an ecosystem home (candidate: a ground-physics-adjacent note).
  **Doctrine-tier → operator's call**, do not edit GROUND_PHYSICS unprompted.
- **The conserved-quantity bridge** — identity = the contrastive residue after nulling the envelope
  = a *measured* instance of GROUND_PHYSICS's "identity conserved, envelope changes." The doctrine's
  metaphor turned literal in the eigenvectors.
- **The sensor→actuator→governance loop** (sk-mcp / CDG-liquid / Cathedral-SBR, iff cone-nulled).
- **Mood Classifier reconciliation** — read `cathedral-and-codex/03_ADRS/ADR_ The Mood Classifier.md`
  and reconcile vs tonight's **contrastive-SBR requirement**; cross-link to the co-adaptation doc.

## Open decisions / loose ends (banked — NOT aims; dispose when convenient)

- sk-mcp branch `session/2026-07-12-anisotropy-instruments` is **un-merged to main** — PR / fast-forward / leave.
- Sept-2024 turns (8267–8289) → SPINE "Roots (corpus archaeology)" — offered, not done.
- **3 files mode `600`** in `incoming-ideas/` (`files (15)/ADR-XXXX-*`, `loose-ends-2026-07-01.md`)
  blocked those 3 from the idea-corpus embed — `chmod g+r`, operator-owned.
- **Un-minted ADR placeholders** (`ADR-XXXX-*`, `ADR-0NN-*`, `ADR-tbd-001`) in `incoming-ideas/` —
  dangling mints / curation debt (double-flagged: also the permission gap above).

## Pointers (the map)

- Law & synthesis: this file. · Sk-mcp research: `docs/research/co-adaptation-longitudinal.md`,
  `docs/map.md`, `docs/SPINE.md`. · Instruments + runbook: `scripts/measure_cone.py`,
  `scripts/probe_axis_poles.py`, `docs/runbooks/anisotropy-instruments.md`.
- Locked experiments: `design-docs/experiments/idea-corpus-latent-axes/` · store `.../idea-corpus-nv4096/`
  · index `design-docs/experiments/README.md`.
- CDG: `ComfyUI-DiffusionGemma/docs/experiments/` (liquid-state note + `experiment.md`).
- Governance: `design-docs/cathedral-and-codex/00_CANON/SPEC-CATH-001_...` + `03_ADRS/ADR_ The Mood Classifier.md`.
- Pilot data: `semantic-chunker/data/tat_metrics/exclamation_analysis_2023-2025.json` + `incoming-ideas/images/`.
