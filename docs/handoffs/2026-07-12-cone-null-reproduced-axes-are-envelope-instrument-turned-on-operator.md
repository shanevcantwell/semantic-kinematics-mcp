# Handoff — 2026-07-12 · cone-null reproduced · the axes are envelope · the instrument turned on the operator

**For the cold next session.** Durable framing is [`docs/SPINE.md`](../SPINE.md); the new research
cluster is [`docs/research/co-adaptation-longitudinal.md`](../research/co-adaptation-longitudinal.md);
live corpus counts are [`docs/generated/CORPUS_STATS.md`](../generated/CORPUS_STATS.md). This file is
the session-continuity note — forward-weighted: what was *established*, what it *means*, what is now
*explorable*.

## TL;DR — the meanings, compressed

1. **Cone-null is now reproducible, not remembered.** `scripts/measure_cone.py` (new) reproduces the
   2026-07-09 ad-hoc baseline *exactly* on N=86,748 non-zero vectors: ‖μ‖=0.555, mean pairwise
   cosine=0.311 ≈ ‖μ‖² (the identity), **centered** participation ratio ≈34 (top-50 = 54.5%).
   **De-meaning alone lifts effective rank 8.1 → 34.1**, removing exactly ‖μ‖²=0.308 of the total
   energy — the single cheap step does most of the nullification. (`map.md`'s "~80×" baseline was an
   error, corrected to ~160×.)

2. **The top axes are envelope, not meaning.** `scripts/probe_axis_poles.py` (new) pulls the pole
   passages of the top-2 centered eigenvectors: **PC1 (14.8%) = export-channel** (verbose Gemini
   prose ↔ terse Claude-Code tool-stubs), **PC2 (5.0%) = register** (abstract ↔ terse-agentic).
   *Semantic topic lives below ~5%.* PC1 is partly a **duplication artifact** — the −pole is a mass
   of byte-identical `[Tool use: Read]` stubs (bottom-200 is 100% `claude_code`). **Corpus hygiene
   (tool-stub filtering) is a precondition, not a nicety** (relates tvi#44). Nullification is a
   *high-pass filter*: it removes common structure to expose distinctive structure.

3. **The methodological spine, re-derived from first principles (operator).** A null estimated from
   the *same data you measure* taints its own common axes — you go blind to exactly the axes that are
   both anisotropy and signal. Resolution: measure **contrastive, not absolute** (z_A − z_B against
   the same null cancels the shared taint); **the null is the experiment.** This is why absolute
   affect died (ADR-SKMCP-0005) and contrastive survives — now derived, not just observed.

4. **New research cluster banked: co-adaptation — the instrument turned on the operator** (mirror of
   Aim 3). Full doc + `map.md` node written this session.
   - **Node A — LLM-accent / reverse entrainment:** register adoption as a **lagged step-response
     with overshoot** around changepoints (release / sysprompt / RBR); *not* a 2-message atom. The
     tell is **migration from conditioned to saturated affect** (entropy collapse), not rate. Pilot:
     `semantic-chunker/.../exclamation_analysis_2023-2025.json` — amplification 1.77×, opening 4.76×,
     and a decile crossover where the user *out-exclaims the receding AI* late (motivating, not
     confirming). Phenomenological anchor (rare first-person data): the operator's "Good!" openers
     began as deliberate forward-pass steering, then **saturated into μ** (common-mode = null at the
     margin) → compulsory-but-functionless ritual. Anisotropy from the inside.
   - **Node B — ideation-variability trajectory:** a reported ~2yr dispersion drop, **encoder
     UNRECOVERED** (reported-not-grounded; step zero = recover the artifact). The model-authored
     "reduced ADHD influence" reading is **firewalled** — high-inference, out of instrument reach, a
     specimen of the completion-bias the program studies; primary confound is channel composition.
   - **Unifying hypothesis:** B may be A's *shadow* — entraining to a bounded register mechanically
     narrows dispersion, needing no cognitive story.

5. **The Cathedral connection: this instrument is the sensor for a governance body.**
   `design-docs/cathedral-and-codex/00_CANON/SPEC-CATH-001` — the Companion architecture's
   **State-Based Resonance** State Vector ("valence, topic, goals") **is** the hyper-personalized
   mood classifier the instrument produces. Sharp contribution: **SBR must be contrastive /
   cone-nulled or it becomes the Emotional Echo Chamber the spec fears (R06)** — the anisotropy work
   is the fix at the measurement layer. Same founding concern (the manipulation-mitigation patent =
   SPINE roots turn 8913): **sk-mcp is the sensor, the Cathedral is the actuator.**

6. **Roots deepened.** Sept-2024 corpus turns (8267–8289) recognized as founding DNA: the anisotropy
   folk-theory (8283 — "repeat fragments → weight those pathways"), the kudzu seed (8267–69 —
   alignment-as-personality → pressuring users), the synthetic-data-recursion worry (8285, now
   mitigated in the Cathedral spec). Candidate addition to SPINE "Roots (corpus archaeology)".

## Current true state (grounded 2026-07-12)

- **Working ground (sk-mcp, `git status -sb`):** branch `main`, even with origin. **All this
  session's work is UNCOMMITTED:** `docs/map.md` (M, 131 ln); untracked `docs/research/co-adaptation-longitudinal.md`
  (150 ln), `scripts/measure_cone.py` (276 ln), `scripts/probe_axis_poles.py` (419 ln).
- **Corpus (nv4096):** main `corpus_4096.jsonl` = 98,293 raw → **87,004 unique** (retry dups,
  documented); Jul-11 delta = 40,457 raw → **40,100 unique**, **disjoint** from main (0 shared
  chunk_ids — genuinely new WSL2-era Claude Code sessions). Generator (`summarize_corpus.py`) already
  dedups; incremental concat/refresh is **tvi#47** (open). `/mnt/storage` symlink is live (tvi#45
  partially materialized).

## Applied uses to explore (the forward stack — reuse-first, ranked)

1. **Mood-classifier MVP — the realistic Codex beachhead.** Build the SBR State Vector on the
   *existing* `analyze_axis_alignment` primitive + the operator's self-null. Cheap · mathematical ·
   contrastive · falsifiable; standalone-useful even if the full Cathedral never ships; and it is
   Branch 1's seed.
2. **Node A prior-art recovery.** `design-docs/incoming-ideas/images/` holds
   `exclamation_entrainment_with_releases.png`, `chart2_who_leads_whom.png`,
   `chart1_inverse_correlation.png`, `quarterly_entrainment.png` (Nov–Dec 2025) — looks like an
   *already-run* release-anchored entrainment analysis. Recover it + find the generating script
   (**may also be Node B's lost encoder**). Reuse-not-build before rebuilding the first cut.
3. **Node B encoder recovery (step zero).** Chase `semantic-chunker/scripts/forensics/structural_fingerprint.py`.
4. **Cathedral bridge doc.** Pin the **contrastive-SBR requirement** into the Cathedral canon
   (`RESEARCH-CATHEDRAL-001_Data-Governance-Subgraph`) so the echo-chamber fix isn't lost on the
   next pass.
5. **Corpus hygiene.** Filter `[Tool use: …]` boilerplate, re-run `probe_axis_poles.py` to test
   whether a *semantic* axis surfaces under the envelope (direct test of the PC1-artifact hypothesis).

## Pending choices for the operator

- **Commit this session's artifacts?** 4 files uncommitted on shared `main` — commit via the git
  subagent on a branch (shared-checkout discipline), or leave as working state.
- **Add the Sept-2024 turns to SPINE Roots archaeology?** Offered, not yet done.

## Pointers

- Framing: `docs/SPINE.md` · Cluster: `docs/research/co-adaptation-longitudinal.md` · Targets: `docs/map.md`
- New scripts: `scripts/measure_cone.py`, `scripts/probe_axis_poles.py`
- Pilot: `semantic-chunker/data/tat_metrics/exclamation_analysis_2023-2025.json` (+ the images dir)
- Governance: `design-docs/cathedral-and-codex/00_CANON/SPEC-CATH-001_Companion-Cognitive-Architecture.md`
- Prior decisions: ADR-SKMCP-0005 (absolute affect falsified), ADR-SKMCP-0006 (generated stats)
