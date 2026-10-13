# Historical static-fusion addendum: lift readout

These are the user's supplied R20261009A_FUSION results, not results from the new notebooks. The TEST cohort contained 3,414 snapshots and 202 positive snapshots. Top 10%, 20% and 30% selected 342, 683 and 1,025 snapshots respectively. Lift is selected precision divided by the cohort positive rate (approximately 5.92%).

| Variant | Approach | Lift at 10% | Lift at 20% | Lift at 30% | TEST AP |
| --- | --- | ---: | ---: | ---: | ---: |
| V0 | Locked count Transformer | 2.866 | 2.079 | 1.781 | 0.1397 |
| V1 | AutoML-only LightGBM | 3.163 | 2.326 | 1.995 | 0.1451 |
| V2 | Frozen representation + AutoML, logistic head | 3.064 | 2.103 | 1.764 | 0.1469 |
| V2c | Frozen representation, logistic head | 2.669 | 2.004 | 1.748 | 0.1375 |
| V3 | Frozen representation + AutoML, LightGBM head | 2.669 | 2.202 | 1.896 | 0.1400 |
| V3c | Frozen representation, LightGBM head | 2.817 | 2.004 | 1.748 | 0.1349 |
| V4 | Warm-start Transformer + static branch | 3.113 | 2.004 | 1.830 | 0.1404 |
| V5 | Warm-start control, zero static input | 2.669 | 2.004 | 1.764 | 0.1344 |

AutoML alone had the highest reported lift at all three capacities. At 10%, V1 captured 64 positives versus 58 for V0; V2 captured 62 and V4 captured 63. Counts here are reconstructed from the supplied recall values and 202 positives. The six additional positives for V1 correspond to approximately 10.3% more positive captures at the same capacity. Rounded lift values give approximately 10.4% relative lift improvement.

The strongest fusion at 10% was V4, but it did not outperform AutoML alone. V2 had the highest AP, illustrating why AP and lift at a particular capacity can choose different models. V2 exceeded its representation-only control at 10% (3.064 versus 2.669); V4 exceeded its continued-training control (3.113 versus 2.669). V3 was worse than its representation-only control at 10% (2.669 versus 2.817), although it was better at 20% and 30%. This pattern does not establish that sequence interactions are needed.

## Limits of the finding

- No lift confidence intervals were supplied. The addendum's AP bootstrap intervals all crossed zero; that does not answer whether lift differences are statistically distinguishable.
- Printed bootstrap deltas have the opposite sign to the variant-minus-V0 AP differences. For example, V2 AP exceeds V0 while the printed delta is negative. Interpret these as V0 minus variant unless the underlying comparison implementation establishes otherwise.
- The now-supplied feature code uses RESP in mapping/pruning, positive-class caps, feature typing and feature selection without a TRAIN restriction. Together with the reported all-split extraction universe, this prevents treating the old AutoML comparisons as clean held-out evidence. Later TRAIN-only scaling cannot undo upstream label use.
- This was the second evaluation of the same TEST cohort. The request to make lift primary followed these results, so this lift readout is retrospective. It is not a newly pre-registered confirmation or independent out-of-time validation.
- V4's implementation adds a static projection to the pooled representation before the head. It does not insert historical AutoML values into the sequence tensor. The old addendum therefore does not test the new temporal AutoML experiment.

Client wording: "The earlier retrospective addendum showed higher fixed-capacity lift from AutoML features, with AutoML alone strongest at the reported 10%, 20% and 30% capacities. Static fusion was promising at 10%, but did not establish an advantage over AutoML alone. We are rebuilding the comparison with raw historical features, TRAIN-only selection and lift-based evaluation; no new model performance result is available yet."
