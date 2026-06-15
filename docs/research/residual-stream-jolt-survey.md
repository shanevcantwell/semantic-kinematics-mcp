# Residual-Stream Jolt Survey

*Scalpel literature survey for token-level "jolt" detection via the residual stream. Compiled 2026-06-15. Every cited source is an arXiv paper read via its abstract; URLs are arxiv.org abstract pages. Each entry is tagged against the project's empirical findings C1–C5 (stated below).*

## Roadmap & empirical lens (as of 2026-06-15)
North star: measure the trajectory of a reasoning model through its **residual stream** (local GGUF models under llama.cpp on the host — NOT Claude's internals, which are inaccessible, and NOT CC reasoning-trace logs, which don't retain thinking tokens). Embedding-API probes (sentence-atom, token-wise prefix/pooling-none) are stepping-stone recon. Sentence-atom magnitude jolt detection is a committed, trustworthy NEGATIVE (PR #32). Findings C1–C5 above are the priors this survey interrogates.

### The empirical lens (priors each source is mapped against)
- **C1.** Cosine/curvature *angles* between successive embedding displacements saturate near π/2 in high dimension → turn-angle/curvature is non-discriminative; magnitude survives where angle dies.
- **C2.** Sentence-pooled displacement *magnitude* does NOT separate human-legible jolts from ordinary real-text motion (embeddinggemma-768; peak 1.43σ vs a measured within-turn null; nothing clears 3σ; top steps land on expository sentences, not punchlines). Atom/scale mismatch, not "no signal."
- **C3.** "Escalation" is NOT a single coherent linear axis in a sentence embedder (union-SVD over an anchor grid: top component ~14%, within-axis coherence 0.10–0.17).
- **C4.** Real-text displacement deltas are anisotropic (cone structure) → nulls must be MEASURED from real text, not assumed isotropic (1/√d).
- **C5.** Working hypothesis: the phenomenon is TOKEN-level (single-token slips/bleeds); sentence pooling averages exactly that away. North star is the generating model's residual stream.

---

## Q1. Residual-stream trajectory analysis: reading hidden-state dynamics across layers and token positions

**Belrose et al. (2023), "Eliciting Latent Predictions from Transformers with the Tuned Lens."** arXiv:2303.08112. https://arxiv.org/abs/2303.08112
Trains an affine probe per block of a frozen model to decode every hidden state into a vocabulary distribution; a refinement of the brittle logit lens. Crucially: the *trajectory of latent predictions across layers* detects malicious inputs with high accuracy. This is the canonical "read the residual stream layer-by-layer as a sequence" instrument and the foundational tool for the rig.
**Lens: extends C1/C5** — establishes the across-layer trajectory (not the across-token embedding trajectory the project has been probing) as the measurable object, and shows it already carries an anomaly signal.

**Yang et al. (2025), "Understanding Aha Moments: from External Observations to Internal Mechanisms."** arXiv:2504.02956. https://arxiv.org/abs/2504.02956
Studies the "aha moment" in large reasoning models from linguistic patterns through latent-space analysis. Finds an internal separation between anthropomorphic/self-reflective characteristics and pure reasoning, and that perceived problem difficulty shifts across layers. Locates *where* a self-correction "jolt" registers internally.
**Lens: extends C5** — the jolt-analog (aha moment) has an internal latent signature distinct from surface tokens; supports moving the atom inside the model.

**Furuya & Tanimura (2026), "Spatiotemporal Hidden-State Dynamics as a Signature of Internal Reasoning in LLMs."** arXiv:2605.01853. https://arxiv.org/abs/2605.01853
Defines **StALT** (Spatiotemporal Amplitude of Latent Transition): a *training-free trajectory statistic* summarizing hidden-state change between adjacent decoding tokens, weighted by within-token layer saliency. Successful reasoning shows broad temporal dynamics with localized layer-wise concentration; StALT separates correct from incorrect trajectories. This is almost exactly the project's "semantic kinematics" idea relocated from the embedding API to the residual stream, with the atom set to the *token* and an explicit layer-weighting.
**Lens: extends C2/C5, refines C1** — magnitude of inter-token latent transition DOES discriminate when measured token-wise inside the model (where C2's sentence-pooled version failed); confirms the atom is the lever.

**Kim, Yoo & Oh (2025), "On the Effect of Uncertainty on Layer-wise Inference Dynamics."** arXiv:2507.06722. https://arxiv.org/abs/2507.06722 (ICML 2025 Actionable Interpretability Workshop)
Uses the Tuned Lens to track layer-wise probability trajectories of the final token across 11 datasets / 5 models. Finds certain and uncertain predictions have *largely aligned* trajectories — both show abrupt confidence jumps at similar layers — so simple uncertainty does NOT visibly perturb inference dynamics (though stronger models may differ).
**Lens: challenges C5 (cautionary)** — a naive "the residual-stream trajectory will visibly lurch when the model is unsure" hypothesis is contradicted for the across-layer *probability* trajectory; the discriminating observable must be subtler than gross trajectory shape. Directly warns against repeating C2's failure one level deeper.

**Liu (2026), "The Spectral Geometry of Thought: Phase Transitions, ... Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason."** arXiv:2604.15350. https://arxiv.org/abs/2604.15350
Across 11 models / 5 families: reasoning vs factual recall shows **spectral phase transitions** in hidden-activation space. Reports a *per-token spectral cascade* (local synchronization decaying with layer distance), **phase-transition signatures aligned with reasoning-step boundaries**, and spectral-alpha correctness prediction at AUC up to 1.000 in late layers *before* the answer is produced. The single most on-thesis paper: a token-level internal observable that punctuates at reasoning-step boundaries — i.e. exactly where a "jolt" should live.
**Lens: extends C5, refines C1** — when angle/magnitude die in high-d, a *spectral* (eigenstructure) observable survives and is step-boundary-aligned; argues the right observable is geometric-spectral, not displacement-magnitude.

**Dumas (2025), "nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers."** arXiv:2511.14465. https://arxiv.org/abs/2511.14465 (NeurIPS 2025 MechInterp Workshop)
Wrapper over NNsight giving one interface for logit lens, patchscope, activation steering, and attention-probability access across 50+ HF model variants. Tooling pointer for Q1/Q5; note it targets HuggingFace, not GGUF.
**Lens: enables C5** (tooling, see Q5 risk).

---

## Q2. Token-level semantic surprise / anomaly from internal states or next-token distributions

**Wang et al. (2025), "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning."** arXiv:2506.01939. https://arxiv.org/abs/2506.01939 (NeurIPS 2025)
Only a small fraction of CoT tokens have high entropy; these **"forking tokens"** act as decision points steering the model into different reasoning pathways. Restricting policy-gradient updates to the top-20% high-entropy tokens matches or beats full updates. The strongest external evidence that the phenomenon of interest is a *sparse, token-localized* event with a clean internal correlate (next-token entropy) — precisely C5, and precisely what sentence pooling destroys.
**Lens: confirms C5, explains C2** — jolts are a high-entropy minority of tokens; pooling over a sentence averages the forking token into ~dozens of low-entropy ones, which is exactly why C2's sentence-atom magnitude saw nothing. Predictive entropy at the token is the candidate observable C2 lacked.

**Yuan et al. (2026), "Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal."** arXiv:2605.09502. https://arxiv.org/abs/2605.09502 (ICML 2026 MechInterp)
A linear probe on hidden states predicts trace correctness at **0.95 AUROC — from the very first reasoning step (0.79)** — while a text-surface classifier reaches only 0.59 and the model *verbalizes* equal confidence on right and wrong traces. Holds across Qwen/Llama/Phi, 1.5B–72B, and RL reasoning models. But four interventions (steering, best-of-N, self-correction, patching) all fail: the signal is a readout, not a lever.
**Lens: strongly confirms C5, sharply challenges the surface approach** — the discriminating signal lives in hidden states and is **invisible in generated text** (0.95 internal vs 0.59 surface). This is the empirical heart of the pivot to the residual stream. Caveat to absorb: the signal is *diagnostic*; expect to *detect* jolts, not necessarily to read a clean magnitude that "causes" them.

**Su et al. (2024), "Unsupervised Real-Time Hallucination Detection based on the Internal States of LLMs" (MIND).** arXiv:2403.06448. https://arxiv.org/abs/2403.06448
MIND uses internal states during inference for real-time, annotation-free hallucination detection, outperforming post-hoc methods. Establishes that per-step internal state is a usable online anomaly channel.
**Lens: extends C5** — online internal-state monitoring is feasible and beats text post-processing.

**Wang et al. (2025), "What are Models Thinking about? ... Hallucinations through Model Inner State Analysis."** arXiv:2502.13490. https://arxiv.org/abs/2502.13490
Splits the forward pass into understanding / query / generation stages and evaluates which internal states reveal hallucination. Useful for choosing *where in the stream* (which stage/layer band) to read.
**Lens: extends C5** — informs layer/stage selection for the observable.

**Yi Liu spectral correctness (2604.15350, see Q1)** and **arithmetic-error probing (Sun et al. 2025, arXiv:2507.12379, https://arxiv.org/abs/2507.12379)** both show lightweight probes on hidden states predict per-step correctness >90%.
**Lens: confirms C5** — the per-token/per-step internal readout is repeatedly the best tracker of a perceived slip; magnitude at the sentence atom (C2) was the wrong observable AND the wrong atom.

*Net Q2 finding:* across hallucination, error-awareness, and entropy lines, the measure that best tracks a *perceived* jolt is a **per-token internal-state probe or token-entropy spike**, not displacement magnitude. This is the cleanest external correction of C2.

---

## Q3. Embedding-space anisotropy — why raw magnitude/cosine baselines mislead

**Ethayarajh (2019), "How Contextual are Contextualized Word Representations? ... Geometry of BERT, ELMo, GPT-2."** arXiv:1909.00512. https://arxiv.org/abs/1909.00512 (EMNLP 2019)
Contextual representations are **not isotropic in any layer**; same-word self-similarity drops in upper layers; <5% of a word's contextual variance is explained by a static embedding. The canonical anisotropy result.
**Lens: confirms/refines C4** — anisotropy is layer-dependent and intrinsic; any null built assuming isotropy (1/√d) is wrong, and the *degree* of wrongness varies by layer — so a measured, **per-layer** null is required, not a single global one.

**Gao et al. (2019), "Representation Degeneration Problem in Training Natural Language Generation Models."** arXiv:1907.12009. https://arxiv.org/abs/1907.12009 (ICLR 2019)
Likelihood training with weight tying pushes learned embeddings into a **narrow cone**, limiting representational power. The mechanistic origin of the cone/anisotropy the project measured directly.
**Lens: confirms C4** — names the exact "cone structure" C4 reports and gives it a training-dynamics cause; the cone is expected, not an artifact of the project's pipeline.

**Diera, Galke & Scherp (2024), "Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search."** arXiv:2411.17538. https://arxiv.org/abs/2411.17538
Low isotropy hurts semantic-inference tasks; Soft-ZCA whitening (controllable isotropy) improves retrieval. A concrete remedy if the rig needs an isotropized comparison space.
**Lens: refines C4** — offers whitening as the lever to convert an anisotropic space into one where Euclidean/cosine baselines mean what they appear to; but note whitening can amplify noise (below).

**Zhang et al. (2024), "Are ID Embeddings Necessary? Whitening Pre-trained Text Embeddings ..." (WhitenRec+).** arXiv:2402.10602. https://arxiv.org/abs/2402.10602
Pretrained text embeddings sit in an anisotropic space with **average pairwise cosine >0.8**; full whitening helps but can break the semantic manifold, motivating *relaxed* whitening.
**Lens: confirms C4, cautions C1** — quantifies how compressed the cone is (cos>0.8 baseline), explaining why raw cosine/angle is near-degenerate (C1); warns that the fix (whitening) is not free.

**Li, Eustratiadis & Kanoulas (2026), "Spectral Tempering for Embedding Compression in Dense Passage Retrieval."** arXiv:2603.19339. https://arxiv.org/abs/2603.19339 (SIGIR 2026)
PCA preserves variance but underuses capacity; whitening enforces isotropy but amplifies heavy-tailed-eigenspectrum noise. Adaptive spectral scaling γ(k) sits between.
**Lens: refines C4** — direct warning that isotropizing a heavy-tailed (anisotropic) spectrum injects noise; the measured-null discipline (C4) is preferable to forcing isotropy.

*Net Q3 finding:* C4 is strongly confirmed and sharpened — anisotropy is intrinsic, layer-dependent, training-induced, and severe (cos>0.8). The literature's "fix" (whitening) is double-edged; the project's instinct to **measure the null from real text rather than assume isotropy is the safer path** and is independently supported.

---

## Q4. "Semantic kinematics" prior art — velocity/curvature of text through embedding/activation space, and the atom question

**Zimmerman (2026), "Semantic Novelty at Scale: Narrative Shape Taxonomy and Readership Prediction in 28,606 Books."** arXiv:2602.20647. https://arxiv.org/abs/2602.20647
Defines "semantic novelty" = cosine distance of each paragraph's SBERT embedding to the running centroid; reduces each book's novelty curve to 16-segment PAA and clusters into 8 narrative archetypes. Reports that **circuitousness has strong raw correlation (ρ=0.41) but is 93% confounded with length** (partial ρ drops to 0.11); "speed" and variance ("volume") are the length-robust predictors.
**Lens: confirms C2, extends C4-discipline** — near-identical method to the project's trajectory work, at the **paragraph atom**, and independently finds the naive path-geometry metric (circuitousness) is a length artifact once controlled. Strong external evidence that coarse-atom trajectory geometry is largely confound, not signal — exactly C2's lesson, with the additional warning to control for length.

**Zimmerman (2026), "Semantic Novelty Trajectories in 80,000 Books: A Cross-Corpus Embedding Analysis."** arXiv:2603.01791. https://arxiv.org/abs/2603.01791
Sentence-transformer paragraph embeddings, running-centroid novelty, trajectory circuitousness (path length / net displacement), and convergent/divergent narrative curves across 80k books. Notably parallel to the project's own ~80k-book vault and "circuitousness"/curvature framing.
**Lens: confirms C1/C2** — uses the same path-length-vs-displacement geometry the project found non-discriminative at fine scale; works only as a *coarse, corpus-aggregate* descriptor, never as a token/sentence jolt detector. Confirms the sentence/paragraph atom is for narrative shape, not jolts.

**Vani, Mellace & Antonucci (2020), "Temporal Embeddings and Transformer Models for Narrative Text Understanding."** arXiv:2003.08811. https://arxiv.org/abs/2003.08811
Character-relationship trajectories via dynamic embeddings over narrative time. Trajectory-through-embedding-space prior art at the entity/coarse atom.
**Lens: extends C2** — another coarse-atom trajectory use; no token-level jolt claim.

**Sanchez-Karhunen et al. (2024), "Interpretation of the Intent Detection Problem as Dynamics in a Low-dimensional Space."** arXiv:2408.02838. https://arxiv.org/abs/2408.02838 (ECAI 2024)
Sentences injected into trained RNNs trace trajectories on a low-dim manifold; the network steers trajectories toward attractor regions aligned with output-layer directions. A dynamical-systems reading of hidden-state trajectories.
**Lens: extends C1/C5** — supports trajectory-as-signal but in *hidden* space with directional structure (attractors), echoing that direction matters when read against task-aligned axes, not as raw high-d angle.

*Net Q4 finding:* C2's sentence/paragraph-atom failure is a **known, reproduced result** — independent corpus-scale work (Zimmerman) finds the same path-geometry metrics are confounded (length) or only coarse-descriptive. No surveyed embedding-space kinematics paper reports token-level jolt detection; the discriminating work (StALT, spectral cascade) all moved into the *activation* space and the *token* atom. The project's C1 (curvature useless in high-d) is consistent with the field's silent abandonment of raw curvature in favor of probes/spectra.

---

## Q5. Practical per-token hidden-state extraction from local models

**Fiotto-Kaufman et al. (2024), "NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals."** arXiv:2407.14561. https://arxiv.org/abs/2407.14561
NNsight extends PyTorch with deferred remote execution to read/intervene on internals of large open-weight (HuggingFace) models via an Intervention Graph. The reference path for per-layer residual-stream access — **but it operates on PyTorch/HF models, not GGUF/llama.cpp.**
**Lens: enables C5 (with caveat)** — the clean residual-stream rig is an HF/transformers path, not the host's current llama.cpp/GGUF path.

**Dumas (2025), nnterp (arXiv:2511.14465, see Q1)** and the implied **TransformerLens** baseline it compares against: both are HF-side. nnterp notes TransformerLens introduces numerical mismatch via manual re-implementation; NNsight preserves exact behavior but lacks standardization.
**Lens: enables C5 (caveat)** — confirms the mature per-token activation toolchain assumes HF weights.

**llama.cpp `/embedding` with `--pooling none`:** no peer-reviewed paper exists (the arXiv "llama.cpp + embeddings" hits are robotics-deployment papers: LiteVLA-Edge arXiv:2603.03380, vla.cpp arXiv:2606.08094 — not interpretability). llama.cpp *does* expose per-token embeddings (pooling none) at the *final* layer via the embedding endpoint, but exposing **per-layer residual-stream activations from a GGUF model under llama.cpp is not a documented, published method** — it requires either patching llama.cpp/ggml to dump intermediate KV/residual tensors or running the same GGUF weights through an HF/transformers loader.
**Lens: this is the central feasibility risk for C5** — see "Recommended rig" risks. (Project-asserted capability of `pooling none` is real for final-layer token vectors; per-layer is an engineering build.)

**On whether sentence-embedders' PRE-pooling token vectors carry token-level signal:** Ethayarajh (1909.00512) shows upper-layer token representations are highly context-specific (low self-similarity), i.e. they *do* carry token-specific content, not just positional/syntactic boilerplate — but they are also the most anisotropic. So pre-pooling token vectors from an embedding model are a *cheap stepping-stone* signal source (they are not dominated by position), with the caveat that they are final-layer-only and anisotropic.
**Lens: refines C4/C5** — pre-pooling token vectors are usable recon (supports the embedding-API stepping-stone), but the generating-model residual stream (multi-layer, decoding-time) is the richer instrument the discriminating papers actually use.

---

## Recommended residual-stream rig (concrete, falsifiable first experiment)

**Setup.** Run a small open-weight reasoning model the host already serves in **HF/transformers** form (not GGUF) for the first build — e.g. a Qwen2.5/Qwen3 or DeepSeek-R1-Distill 1.5B–8B, the exact family StALT, Hidden-Error-Awareness, and Spectral-Geometry all validate on. Capture, at **every decoding token**, the residual-stream hidden state at a band of **late-middle layers** (the layers where Hidden-Error-Awareness and spectral correctness peak). Atom = **token**, not sentence (the C2 correction).

**Observable (primary).** Per-token **inter-token latent transition magnitude weighted by layer saliency** — i.e. reproduce **StALT** (2605.01853) as the first instrument, since it is training-free and directly the project's kinematics idea moved to the residual stream. Secondary observables to log in the same pass: (a) per-token **next-token entropy** (the forking-token signal, 2506.01939); (b) a **linear correctness/anomaly probe** on the hidden state (0.95-AUROC line, 2605.09502); (c) optional **spectral-alpha per token** (2604.15350) if StALT under-discriminates.

**Measured null (C4-compliant).** Build the null from **real decoding trajectories of the same model on ordinary, jolt-free prompts** — resample inter-token transition magnitudes *per layer* to preserve anisotropy/cone structure. Do NOT use a 1/√d isotropic null and do NOT whiten as a first move (Q3 warns whitening injects heavy-tail noise).

**What counts as detecting a jolt.** On a held-out set of prompts containing human-legible register shifts / self-corrections / "aha" boundaries, the observable at the annotated jolt token clears the **measured per-layer null at ≥3σ** (the project's own committed bar), AND the peak token co-locates with the human-annotated jolt above chance (e.g. AUROC > 0.7 for jolt-token localization). A clean win additionally reproduces the step-boundary alignment of 2604.15350. Falsification = same flat result as C2 but now at the token atom — which would localize the failure to "not magnitude" rather than "not the atom," routing to the entropy/probe/spectral observables.

**Top risks.**
1. **GGUF/llama.cpp activation access (highest).** Per-layer residual-stream extraction is not a published llama.cpp capability; `--pooling none` gives only final-layer per-token embeddings. The rig's first build should run HF weights (NNsight/nnterp), accepting that this diverges from the host's llama.cpp serving path; bridging back to GGUF is a separate ggml-patching effort.
2. **Anisotropy contaminates the null (C4).** If the null isn't measured per-layer on real text, the cone (cos>0.8, Gao 2019 / Ethayarajh 2019) will make any magnitude look either always-large or always-small. This is the same trap C2/C4 already identified, one level deeper.
3. **Atom/observable confound repeats (C2/Q1-caution).** Kim et al. (2507.06722) show the across-layer *probability* trajectory does NOT visibly move with uncertainty; a naive "trajectory lurches at the jolt" hypothesis may fail again. Mitigate by carrying the entropy and linear-probe observables from the start, since those are the measures with demonstrated per-token discrimination — magnitude alone may again be the wrong observable even at the right atom.

---

## Dead ends confirmed
- **Claude weight/activation introspection** — not available to this project; Claude's internals are inaccessible. *Project-asserted; no public mechanism exists to read Claude residual streams, consistent with Anthropic's closed-weight posture.*
- **CC reasoning-trace logs as a residual-stream proxy** — the roadmap asserts CC logs do not retain thinking tokens. *Project-asserted; not independently corroborated from public docs in this survey.* Either way, even if retained, thinking-token *text* is the surface signal that Hidden-Error-Awareness (2605.09502) shows is the weak channel (0.59 vs 0.95) — so the trace text would be the wrong instrument regardless of retention.

---

## Highest-value contradiction to surface
The literature does **not** contradict C1, C3, C4, or the C2 *negative* — all are confirmed or sharpened. The one place external work pushes back on a project *instinct*: **magnitude is not dead, it was mis-atomed.** StALT (2605.01853) shows inter-token *latent* transition magnitude DOES discriminate reasoning quality when measured token-wise inside the model with layer weighting — the same quantity that failed at the sentence atom (C2). So C2 should be read narrowly ("sentence-pooled magnitude fails"), not broadly ("magnitude fails"). The complementary caution: Kim et al. (2507.06722) show the related across-*layer* probability trajectory is flat to uncertainty — so "magnitude works token-wise" is not automatic; it is observable-specific. The safe synthesis the field supports: **token atom + measured per-layer null + carry entropy/probe observables alongside magnitude.**
