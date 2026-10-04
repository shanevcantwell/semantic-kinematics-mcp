# semantic-kinematics-mcp — Loose Ends Ledger

This file is the SKM repo's compendium for decisions that are real but sub-ADR: operational
pauses, timing calls, provisional postures that will retire when a trigger fires. Each entry
uses the standard loose-ends format; graduation or retirement is noted inline when it occurs.

---

## 2026-06-24 — Affect-geometry / bearing track paused pending nv-embed-v2 promotion

**Category:** provisional
**Related ADR:** ADR-SKM-0004

**Graduation trigger:** nv-embed-v2 served via llauncher (#155) → unpause; run cross-embedder
Spike A; the *finding* graduates per the ADR's open question, this pause just retires.

### Context
Session close, slammed. The bearing/affect-gate track is suspended on the conclusion that
current-embedder results are embedder-captive (see ADR-SKM-0004). Suspended, **not killed** —
good data kept, nothing discarded.

### Decision
Park Spike-A-on-current-tooling and the always-unacceptable-gate exploration until nv-embed-v2
lands. No new affect-geometry measurement on embeddinggemma in the interim.

### Why Not an ADR?
- [x] Hard to reverse? → No — unpause the moment nv-embed is served; the *substrate choice* is the
  ADR, this is only its timing.
- [ ] Surprising without context? → No — it's the direct operational consequence of the ADR.
- [x] Real trade-off? → No trade-off beyond timing; provisional by construction.

### Implementation Notes
- [ ] Does anything on the **infra** track (Spike C tokenization, Dest-1 vault mapping) have
  independent value worth running during the pause, or does it also wait? — **Resolution trigger:**
  first session back; decide per whether vault-mapping pays off without the bearing instrument.
