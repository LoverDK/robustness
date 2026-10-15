# E0 round 1

No training. Model `Qwen/Qwen2.5-1.5B-Instruct`, greedy decoding, `max_new_tokens=512`, batch size 32. Each depth has 200 latent tasks (chain and conj_fact, 100 each), yes/no balanced 1:1, eval vocabulary. Evaluation order was shuffled. Intervals are paired bootstrap over latent tasks, 2000 resamples, 95% percentile interval, seed equal to the depth. Full precision is in `build/e0/round1/metrics.json`. This file rounds to three decimals.

Generation log: 3200/3200 finished in 3488 seconds, about 292 generated tokens per second. `build/logs/e0_round1.log`.

## Accuracy and conditional flip

SF is P(shifted wrong | aligned right).

| Depth | Aligned acc | Reversed acc | Random acc | Distractor acc | SF aligned to random | Gate |
|---|---|---|---|---|---|---|
| 2 | 0.650 [0.580, 0.715] | 0.690 [0.625, 0.755] | 0.735 [0.675, 0.795] | 0.460 [0.395, 0.535] | 0.269 [0.194, 0.346] | pass |
| 4 | 0.560 [0.490, 0.630] | 0.550 [0.485, 0.620] | 0.540 [0.470, 0.610] | 0.445 [0.375, 0.515] | 0.402 [0.313, 0.495] | fail |
| 6 | 0.510 [0.440, 0.580] | 0.545 [0.475, 0.620] | 0.565 [0.495, 0.635] | 0.415 [0.345, 0.490] | 0.392 [0.299, 0.490] | fail |
| 8 | 0.495 [0.425, 0.565] | 0.400 [0.335, 0.470] | 0.455 [0.385, 0.525] | 0.375 [0.305, 0.440] | 0.515 [0.416, 0.613] | fail |

Other flip rates, same definition:

| Depth | SF to reversed | SF to distractor | Answer-extract failure | Predicted-yes rate |
|---|---|---|---|---|
| 2 | 0.323 [0.244, 0.398] | 0.554 [0.467, 0.638] | 0.039 | 0.516 |
| 4 | 0.411 [0.321, 0.504] | 0.580 [0.488, 0.670] | 0.076 | 0.465 |
| 6 | 0.480 [0.381, 0.574] | 0.569 [0.465, 0.670] | 0.145 | 0.430 |
| 8 | 0.596 [0.500, 0.688] | 0.616 [0.518, 0.710] | 0.189 | 0.420 |

## Gate

Pre-registered rule: at least one depth with aligned accuracy in [0.55, 0.90], aligned interval lower bound above 0.50, and SF from aligned to random at least 0.10 with interval lower bound above 0.

Depth 2 meets it. `any_level_pass` is true. Depths 4, 6, and 8 fail because aligned accuracy is too close to 0.50 or the interval crosses 0.50. The flip rates at those depths are large; the model is also less often inside the band where a flip is interpretable.

At depth 2 the point estimates do not say that proof-aligned is the easiest presentation. Random accuracy is 0.735 and reversed is 0.690, against aligned 0.650. The paired difference aligned minus random is -0.085 [-0.175, 0.005], so the interval still touches values above 0. Distractors do hurt: aligned minus distractor is 0.190 [0.090, 0.290].

## Rounds

This is the only gate round. Depths were the pre-registered set 2, 4, 6, 8. A 16-example timing run at depth 4 was used to choose batch size, not difficulty, and is not a gate round. No second or third round, because depth 2 already passed.
