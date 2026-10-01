# Knowledge Graphs for Healthcare: rehabilitation recommendation experiment implementation notes

## Experimental formulation

The rehabilitation recommendation experiment constructs temporally ordered
multi-protocol prescription episodes from the Eodyne data. Episodes are formed
when consecutive prescription records for a patient are separated by more than
60 seconds. Patients with at least four episodes are eligible for the
three-history-window experiment.

Targets are restricted to 2024 development and 2025 test windows. The 2024
development targets define a frozen candidate vocabulary of 51 protocols. The
2024 development windows are sorted chronologically and split 80:20 into
training and validation sets. The 2025 windows are held out for final testing.

## Model input and output

| Model                             | Input                                                                       | Output           |
| --------------------------------- | --------------------------------------------------------------------------- | ---------------- |
| Popularity                        | Protocol frequencies in 2024 training targets                               | Ranked protocols |
| Previous Episode / Recent-History | Immediately preceding prescription history with popularity fallback         | Ranked protocols |
| GRU                               | Three 51-dimensional multi-hot episode vectors                              | 51 logits        |
| LSTM                              | Three 51-dimensional multi-hot episode vectors                              | 51 logits        |
| Transformer                       | Same 3 x 51 sequence, projected to 32 dimensions with positional embeddings | 51 logits        |
| TGN-3 BiMem                       | Three temporally ordered prescription episodes                              | 51 scores        |
| TGN-All BiMem                     | All available pre-target prescription episodes                              | 51 scores        |

For TGN-3 and TGN-All, an event represents a patient--protocol interaction
associated with the prescription episode timestamp. Multiple protocols in one
episode are processed simultaneously.

## TGN architecture

The same `TGN3BiMem` architecture is used for TGN-3 and TGN-All. Patients
start each prediction window from a shared trainable initial state; no
patient-ID embedding is used. Each protocol has a trainable 32-dimensional
base representation.

Elapsed time between prescription episodes is measured in days. A learned
linear layer maps elapsed time to 16 dimensions, followed by cosine activation.

For each protocol in an episode, the message input concatenates:

1. current patient memory;
2. protocol base representation;
3. previous protocol temporal memory;
4. 16-dimensional time encoding.

A protocol-to-patient MLP produces patient messages. These messages are mean
aggregated across protocols in the episode. A separate patient-to-protocol MLP
produces protocol messages. Patient and protocol memories are updated with
separate `GRUCell` units.

Candidate scoring combines:

1. patient memory;
2. candidate protocol representation (base + temporal state);
3. element-wise patient--candidate interaction;
4. absolute patient--candidate difference.

The concatenated representation is passed through an MLP to produce one score
for each candidate protocol.

## History definitions

**TGN-3:** exactly the three preceding prescription episodes are processed.

**TGN-All:** every prescription episode preceding the target for the patient
is processed chronologically. The target episode itself is never processed
before prediction.

Both models initialise memory at the start of each prediction window. Thus,
the implementation is a window-based temporal graph-memory recommender rather
than a continuously persistent global TGN.

## Optimisation

The final TGN-All implementation uses PyTorch, `BCEWithLogitsLoss`, Adam,
learning rate 0.001, weight decay 1e-4, dropout 0.2, memory dimension 32,
time-encoding dimension 16, maximum 100 epochs, patience 10, gradient clipping
at 1.0, and random seed 42.

TGN-All model selection is based on validation NDCG@5. The selected checkpoint
is then used for the held-out 2025 evaluation. The TGN-3 experiment in the
notebook uses validation loss for checkpoint selection.

## Evaluation

Predicted logits are converted to probabilities using sigmoid and ranked in
descending order. Metrics are calculated per prediction window and
macro-averaged at K = 1, 3, 5 and 10:

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- HitRate@K
- MRR@K

The final recommendation experiment contains 165 eligible patients, 1,152
prescription episodes, 185 development windows and 472 held-out 2025 test
windows.

## Behavioural endpoint

The secondary behavioural analysis examines protocols played in subsequent
rehabilitation sessions. Observable engagement is defined as `score > 0`.
The observation period is censored at the next prescription where applicable.
`CLOSED` is evaluated separately as a completion-based sensitivity analysis.

A positive score represents observable recorded interaction/performance and does
not establish adherence, therapeutic effectiveness, safety or clinical
appropriateness.

## Patient-cluster bootstrap

Paired patient-cluster bootstrap comparisons use 10,000 replicates and seed 42.
Patients are sampled with replacement and all windows belonging to a sampled
patient are retained together. The 2.5th and 97.5th percentiles form the
reported 95% bootstrap interval.

## Data scope

The recommendation experiment models patient--protocol
interaction history rather than diagnosis-conditioned rehabilitation planning.

The final output layer uses a frozen 51-protocol vocabulary derived from the
2024 development data. New protocols outside this vocabulary cannot be directly
recommended by the identifier-based output layer.

## Implementation boundary

Neo4j/Cypher is used for graph/data preparation. Model training and ranking
evaluation are performed outside Neo4j using Python/PyTorch with pandas and
NumPy.
