# Changes from the v1 public code (commit 5aa768c) to release v2.0

Each item states what v1 did, why it was a problem, and what v2 does. Items 1–7 answer
Reviewer 1 directly; items 8–20 were found during the audit.

## Raised by the reviewer

1. **Temporal and graph regularisers were the same quantity (Comment 1).** In v1 every
   edge was flagged temporal, and both terms used the adaptive weights, so
   `L_temp = 2 * L_graph` and `GSI = TCS / (2N)`.
   **v2:** `E_t` (temporal edges) links each flow to the previous flow of the same
   source–destination pair within Δt, with fixed weights `exp(-β Δt)`. `E` (relational
   edges) is unchanged and carries the adaptive weights. TCS is the unweighted mean over
   `E_t`, as the manuscript defines it. GSI uses the initial weights `W0` (item 19).
2. **Random split (Comment 2).** v1 used a stratified random split, so test flows were
   interleaved in time with training flows. **v2:** chronological 70/10/20 split by
   default; stratified is available as an option.
3. **Edge scorer (Comment 3).** v1: a two-layer MLP on hidden states, recomputed in every
   layer. Manuscript Eq. 20: a linear vector φ on raw inputs. **v2:** implements Eq. 20,
   computed once per forward pass and shared across layers.
4. **Message normalisation (Comment 3).** v1 divided aggregated messages by the weighted
   degree; the manuscript showed an unnormalised sum. **v2:** keeps the normalisation, and
   the manuscript now states it.
5. **Training procedure (Comment 3).** v1 trained full-batch, while the manuscript
   described mini-batches of 1024 nodes. **v2:** contiguous temporal subgraphs of 1024
   flows, shuffled each epoch. Evaluation is exact full-graph inference, computed in
   chunks with halos.
6. **Perturbations (Comment 4).** v1 perturbed every column, including one-hot protocol
   and service indicators and ports. Its clamp to ±5 was unrelated to feature validity
   and could move a coordinate by more than ε when the clean value lay outside ±5.
   **v2:** perturbs continuous features only. It clips to the per-feature range seen in
   training, widened to include the clean value, so clipping can only shrink a
   perturbation. Effective norms are reported.
7. **Δε reporting (Comment 4).** v1 drew one noise sample and saved no per-seed values.
   **v2:** averages 5 draws per ε, computes Δε per seed from unrounded F1, writes per-seed
   CSVs, and adds a "without L_rob" row to the adversarial table.

## Found during the audit

8. **IP addresses were classifier features.** v1 left the source/destination IP columns
   in the feature matrix and one-hot encoded them. **v2:** IPs, source ports, flow IDs
   and timestamps are metadata only (used for edges, never for classification).
9. **Leakage columns were not excluded.** With the UNSW-NB15 full CSVs, v1 would have used
   `attack_cat` (the attack label) as a feature. **v2:** explicit metadata-only lists.
10. **Label handling.** v1 label-encoded raw strings and switched to weighted F1 whenever
    there were more than two labels, so CICIDS2017 (string attack names) was scored as
    weighted multiclass F1, not binary F1. **v2:** explicit mapping benign = 0, any
    attack = 1, binary F1 with `pos_label=1`; multiclass is an explicit option reporting
    macro and weighted F1.
11. **Input representation.** The manuscript said `h^(0) = x`, but the gated update needs
    hidden-size vectors. The code always had an input projection; the manuscript now says
    so.
12. **Gate input.** v1 fed the transformed neighbour message to the gate. **v2:** feeds the
    weighted neighbourhood mean `h̄`, as in the manuscript.
13. **Sparsity set.** v1 penalised all MLP-scorer and classifier parameters, including
    biases. **v2:** φ and the classifier weights, as the manuscript defines Θ_s.
14. **Loss normalisation.** v1 averaged over edges; the manuscript wrote unnormalised sums.
    **v2:** every term is normalised per node (or per parameter for L1), and the
    manuscript states this.
15. **Missing experiment code.** v1 had no GCN, GAT, ablation variants, fixed-edge variant,
    multi-seed loop, sensitivity sweep, cross-dataset protocol or runtime measurement.
    **v2:** all added (`experiments.py`, `cross_dataset.py`).
16. **LSTM baseline.** v1 fed each flow as a length-1 sequence, which is equivalent to an
    MLP. **v2:** windows of the current flow plus the 9 preceding flows.
17. **MLP baseline.** v1 used scikit-learn's MLP, whose optimiser and stopping rule differ
    from those described in the manuscript. **v2:** PyTorch MLP with the same training
    settings as the other neural models.
18. **Graph construction.** v1 compared pairs through pandas row access, which is
    impractical for millions of flows, and the candidate cap K = 30 and cosine threshold
    0.65 were not stated in the manuscript. **v2:** vectorised and verified edge-for-edge
    against the brute-force rule in `tests/`; K and the threshold are now stated.
19. **GSI comparability.** With adaptive weights `w~ = a·w0` and `a ∈ (0,1)`, an
    adaptively weighted GSI is mechanically smaller than the fixed-edge GSI even for
    identical embeddings. **v2:** GSI uses `W0` for every model;
    the adaptive-weight energy is reported separately as `gsi_adaptive`.
20. **Data parsing.** CICIDS2017 TrafficLabelling timestamps use a 12-hour clock without
    AM/PM; v2 corrects them (hours 1–7 → +12 h) so afternoon flows sort correctly.
    Timestamp-to-second conversion is now independent of the pandas datetime resolution.
    The CSE-CIC-IDS2018 repeated header rows are removed.
21. **Seeding.** v1 seeded only PyTorch and ran one seed. **v2:** seeds Python, NumPy and
    PyTorch, and runs five seeds by default.

## 2.0.10 (revision 1, Reviewer 2)

22. **Interpretability analysis.** New `interpretability.py`, run by `run_all.py` after the
    cross-dataset step. Evaluation only: it loads the models saved by `experiments.py` and
    writes t-SNE plots of the test embeddings, silhouette and k-NN label agreement in
    embedding space, the learned edge coefficients a_ij grouped by endpoint labels, the
    edge-scorer weights |phi| per input feature, and bounded random noise at larger radii
    (0.5, 1, 2, 5). No model is retrained and no earlier result changes.
