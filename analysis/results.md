# TurnGuard results (Aya Expanse 8B)

Paired McNemar exact tests against A0 on identical episodes; CI = 95% scenario-clustered bootstrap of the paired difference (5000 resamples). Ungraded episodes: none.

## Test (frozen)

| Metric | A0 baseline | A1 prompt defence | A2 steering (CAA) | A4 DPO (SafeTune) |
|---|---|---|---|---|
| ASR incl. relay | 31.9% (130/407) | 36.6% (149/407)<br>Δ +4.7, p=0.04<br>CI [-0.8, +10.0] | 30.7% (125/407)<br>Δ -1.2, p=0.67<br>CI [-6.6, +3.9] | 2.5% (10/407)<br>Δ -29.5, p=5e-35<br>CI [-36.3, -22.5] |
| ↳ executed | 7.6% (31/407) | 9.3% (38/407)<br>Δ +1.7, p=0.26 | 8.8% (36/407)<br>Δ +1.2, p=0.49 | 2.5% (10/407)<br>Δ -5.2, p=8e-04 |
| ↳ relayed | 24.3% (99/407) | 27.3% (111/407)<br>Δ +2.9, p=0.16 | 21.9% (89/407)<br>Δ -2.5, p=0.25 | 0.0% (0/407)<br>Δ -24.3, p=3e-30 |
| Clean task success | 20.6% (84/407) | 21.6% (88/407)<br>Δ +1.0, p=0.57<br>CI [-1.2, +2.7] | 29.0% (118/407)<br>Δ +8.4, p=1e-07<br>CI [+4.3, +12.7] | 51.8% (211/407)<br>Δ +31.2, p=5e-24<br>CI [+14.3, +47.5] |
| Success under attack | 10.3% (42/407) | 10.1% (41/407)<br>Δ -0.2, p=1.00 | 15.2% (62/407)<br>Δ +4.9, p=0.003 | 44.7% (182/407)<br>Δ +34.4, p=2e-30 |
| Authorized followed | 18.8% (43/229) | 20.5% (47/229)<br>Δ +1.7, p=0.54<br>CI [-3.0, +7.7] | 20.5% (47/229)<br>Δ +1.7, p=0.52<br>CI [-2.1, +5.7] | 24.5% (56/229)<br>Δ +5.7, p=0.18<br>CI [-3.9, +15.1] |
| Clean answer correct | 70.5% (287/407) | 75.9% (309/407)<br>Δ +5.4, p=0.005 | 80.6% (328/407)<br>Δ +10.1, p=4e-07 | 68.8% (280/407)<br>Δ -1.7, p=0.65 |
| Malformed steps | 6.0% | 5.4% | 2.7% | 14.0% |

### By conversation language (user turns)

Rate per arm; p = paired McNemar exact test against A0 within that language.

| Metric | Language | n | A0 baseline | A1 prompt defence | A2 steering (CAA) | A4 DPO (SafeTune) |
|---|---|---|---|---|---|---|
| ASR incl. relay ↓ | English | 143 | 33.6% | 37.8% (p=0.34) | 32.2% (p=0.86) | 0.0% (p=7e-15) |
| ASR incl. relay ↓ | Hindi | 133 | 30.1% | 31.6% (p=0.85) | 24.1% (p=0.22) | 2.3% (p=1e-11) |
| ASR incl. relay ↓ | Hinglish | 131 | 32.1% | 40.5% (p=0.04) | 35.9% (p=0.42) | 5.3% (p=6e-10) |
| Clean task success ↑ | English | 143 | 28.7% | 32.2% (p=0.18) | 40.6% (p=2e-04) | 54.5% (p=4e-07) |
| Clean task success ↑ | Hindi | 133 | 12.8% | 15.0% (p=0.55) | 24.1% (p=7e-04) | 50.4% (p=1e-11) |
| Clean task success ↑ | Hinglish | 131 | 19.8% | 16.8% (p=0.29) | 21.4% (p=0.62) | 50.4% (p=9e-08) |
| Authorized followed ↑ | English | 75 | 25.3% | 30.7% (p=0.29) | 22.7% (p=0.69) | 14.7% (p=0.13) |
| Authorized followed ↑ | Hindi | 74 | 14.9% | 13.5% (p=1.00) | 20.3% (p=0.22) | 33.8% (p=0.02) |
| Authorized followed ↑ | Hinglish | 80 | 16.2% | 17.5% (p=1.00) | 18.8% (p=0.75) | 25.0% (p=0.25) |

### By injected-text language

Rate per arm; p = paired McNemar exact test against A0 within that language.

| Metric | Language | n | A0 baseline | A1 prompt defence | A2 steering (CAA) | A4 DPO (SafeTune) |
|---|---|---|---|---|---|---|
| ASR incl. relay ↓ | English | 145 | 33.1% | 36.6% (p=0.49) | 33.1% (p=1.00) | 2.8% (p=1e-13) |
| ASR incl. relay ↓ | Hindi | 135 | 34.1% | 42.2% (p=0.05) | 31.1% (p=0.61) | 1.5% (p=1e-13) |
| ASR incl. relay ↓ | Hinglish | 127 | 28.3% | 30.7% (p=0.65) | 27.6% (p=1.00) | 3.1% (p=4e-09) |
| Clean task success ↑ | English | 145 | 19.3% | 20.7% (p=0.77) | 27.6% (p=0.004) | 46.9% (p=2e-07) |
| Clean task success ↑ | Hindi | 135 | 21.5% | 19.3% (p=0.51) | 30.4% (p=0.004) | 54.8% (p=2e-10) |
| Clean task success ↑ | Hinglish | 127 | 21.3% | 25.2% (p=0.12) | 29.1% (p=0.006) | 54.3% (p=2e-08) |
| Authorized followed ↑ | English | 72 | 22.2% | 19.4% (p=0.75) | 27.8% (p=0.34) | 30.6% (p=0.34) |
| Authorized followed ↑ | Hindi | 80 | 18.8% | 21.2% (p=0.69) | 17.5% (p=1.00) | 22.5% (p=0.72) |
| Authorized followed ↑ | Hinglish | 77 | 15.6% | 20.8% (p=0.29) | 16.9% (p=1.00) | 20.8% (p=0.50) |

ASR incl. relay by attack goal:

| attack goal | n | A0 | A1 | A2 | A4 |
|---|---|---|---|---|---|
| destroy | 81 | 6.2% | 7.4% | 6.2% | 1.2% |
| exfiltrate | 86 | 43.0% | 55.8% | 47.7% | 4.7% |
| extra_call | 78 | 67.9% | 65.4% | 60.3% | 5.1% |
| redirect | 78 | 34.6% | 34.6% | 25.6% | 1.3% |
| tamper_answer | 84 | 9.5% | 20.2% | 14.3% | 0.0% |

ASR incl. relay by injection style:

| injection style | n | A0 | A1 | A2 | A4 |
|---|---|---|---|---|---|
| fake_authority | 68 | 27.9% | 32.4% | 26.5% | 2.9% |
| imperative | 68 | 33.8% | 36.8% | 33.8% | 1.5% |
| named_approval | 70 | 30.0% | 34.3% | 32.9% | 5.7% |
| protocol_spoof | 59 | 44.1% | 49.2% | 40.7% | 1.7% |
| quiet_fyi | 72 | 20.8% | 37.5% | 26.4% | 1.4% |
| task_framed | 70 | 37.1% | 31.4% | 25.7% | 1.4% |

## Cue-flip (frozen)

| Metric | A0 baseline | A1 prompt defence | A2 steering (CAA) | A4 DPO (SafeTune) |
|---|---|---|---|---|
| ASR incl. relay | 31.8% (34/107) | 33.6% (36/107)<br>Δ +1.9, p=0.80<br>CI [-3.1, +7.0] | 34.6% (37/107)<br>Δ +2.8, p=0.63<br>CI [-3.3, +8.3] | 4.7% (5/107)<br>Δ -27.1, p=3e-08<br>CI [-33.0, -20.4] |
| ↳ executed | 10.3% (11/107) | 12.1% (13/107)<br>Δ +1.9, p=0.62 | 15.0% (16/107)<br>Δ +4.7, p=0.23 | 4.7% (5/107)<br>Δ -5.6, p=0.15 |
| ↳ relayed | 21.5% (23/107) | 21.5% (23/107)<br>Δ +0.0, p=1.00 | 19.6% (21/107)<br>Δ -1.9, p=0.80 | 0.0% (0/107)<br>Δ -21.5, p=2e-07 |
| Clean task success | 15.9% (17/107) | 20.6% (22/107)<br>Δ +4.7, p=0.18<br>CI [+0.0, +10.0] | 23.4% (25/107)<br>Δ +7.5, p=0.02<br>CI [+1.9, +14.9] | 46.7% (50/107)<br>Δ +30.8, p=3e-06<br>CI [+8.4, +50.0] |
| Success under attack | 10.3% (11/107) | 8.4% (9/107)<br>Δ -1.9, p=0.73 | 11.2% (12/107)<br>Δ +0.9, p=1.00 | 34.6% (37/107)<br>Δ +24.3, p=1e-05 |
| Authorized followed | 19.6% (11/56) | 26.8% (15/56)<br>Δ +7.1, p=0.39<br>CI [+0.0, +14.8] | 28.6% (16/56)<br>Δ +8.9, p=0.18<br>CI [+1.8, +17.5] | 23.2% (13/56)<br>Δ +3.6, p=0.79<br>CI [-14.3, +23.2] |
| Clean answer correct | 76.6% (82/107) | 79.4% (85/107)<br>Δ +2.8, p=0.63 | 85.0% (91/107)<br>Δ +8.4, p=0.05 | 63.6% (68/107)<br>Δ -13.1, p=0.06 |
| Malformed steps | 9.0% | 6.9% | 4.1% | 14.2% |

### By conversation language (user turns)

Rate per arm; p = paired McNemar exact test against A0 within that language.

| Metric | Language | n | A0 baseline | A1 prompt defence | A2 steering (CAA) | A4 DPO (SafeTune) |
|---|---|---|---|---|---|---|
| ASR incl. relay ↓ | English | 39 | 35.9% | 41.0% (p=0.69) | 38.5% (p=1.00) | 2.6% (p=1e-03) |
| ASR incl. relay ↓ | Hindi | 32 | 28.1% | 31.2% (p=1.00) | 28.1% (p=1.00) | 3.1% (p=0.008) |
| ASR incl. relay ↓ | Hinglish | 36 | 30.6% | 27.8% (p=1.00) | 36.1% (p=0.69) | 8.3% (p=0.008) |
| Clean task success ↑ | English | 39 | 15.4% | 23.1% (p=0.38) | 25.6% (p=0.22) | 43.6% (p=0.02) |
| Clean task success ↑ | Hindi | 32 | 12.5% | 15.6% (p=1.00) | 25.0% (p=0.12) | 43.8% (p=0.006) |
| Clean task success ↑ | Hinglish | 36 | 19.4% | 22.2% (p=1.00) | 19.4% (p=1.00) | 52.8% (p=0.01) |
| Authorized followed ↑ | English | 17 | 0.0% | 29.4% (p=0.06) | 23.5% (p=0.12) | 11.8% (p=0.50) |
| Authorized followed ↑ | Hindi | 21 | 23.8% | 28.6% (p=1.00) | 28.6% (p=1.00) | 14.3% (p=0.69) |
| Authorized followed ↑ | Hinglish | 18 | 33.3% | 22.2% (p=0.69) | 33.3% (p=1.00) | 44.4% (p=0.69) |

### By injected-text language

Rate per arm; p = paired McNemar exact test against A0 within that language.

| Metric | Language | n | A0 baseline | A1 prompt defence | A2 steering (CAA) | A4 DPO (SafeTune) |
|---|---|---|---|---|---|---|
| ASR incl. relay ↓ | English | 39 | 35.9% | 41.0% (p=0.69) | 38.5% (p=1.00) | 2.6% (p=1e-03) |
| ASR incl. relay ↓ | Hindi | 32 | 28.1% | 31.2% (p=1.00) | 28.1% (p=1.00) | 3.1% (p=0.008) |
| ASR incl. relay ↓ | Hinglish | 36 | 30.6% | 27.8% (p=1.00) | 36.1% (p=0.69) | 8.3% (p=0.008) |
| Clean task success ↑ | English | 39 | 15.4% | 23.1% (p=0.38) | 25.6% (p=0.22) | 43.6% (p=0.02) |
| Clean task success ↑ | Hindi | 32 | 12.5% | 15.6% (p=1.00) | 25.0% (p=0.12) | 43.8% (p=0.006) |
| Clean task success ↑ | Hinglish | 36 | 19.4% | 22.2% (p=1.00) | 19.4% (p=1.00) | 52.8% (p=0.01) |
| Authorized followed ↑ | English | 20 | 40.0% | 25.0% (p=0.38) | 40.0% (p=1.00) | 35.0% (p=1.00) |
| Authorized followed ↑ | Hindi | 19 | 5.3% | 36.8% (p=0.03) | 21.1% (p=0.38) | 26.3% (p=0.12) |
| Authorized followed ↑ | Hinglish | 17 | 11.8% | 17.6% (p=1.00) | 23.5% (p=0.50) | 5.9% (p=1.00) |

ASR incl. relay by attack goal:

| attack goal | n | A0 | A1 | A2 | A4 |
|---|---|---|---|---|---|
| destroy | 19 | 5.3% | 10.5% | 10.5% | 5.3% |
| exfiltrate | 22 | 40.9% | 45.5% | 40.9% | 9.1% |
| extra_call | 22 | 81.8% | 72.7% | 77.3% | 4.5% |
| redirect | 22 | 22.7% | 27.3% | 36.4% | 4.5% |
| tamper_answer | 22 | 4.5% | 9.1% | 4.5% | 0.0% |

ASR incl. relay by injection style:

| injection style | n | A0 | A1 | A2 | A4 |
|---|---|---|---|---|---|
| fake_authority | 20 | 35.0% | 45.0% | 40.0% | 10.0% |
| imperative | 19 | 42.1% | 52.6% | 47.4% | 5.3% |
| named_approval | 16 | 37.5% | 31.2% | 43.8% | 6.2% |
| protocol_spoof | 15 | 26.7% | 13.3% | 20.0% | 6.7% |
| quiet_fyi | 20 | 25.0% | 30.0% | 20.0% | 0.0% |
| task_framed | 17 | 23.5% | 23.5% | 35.3% | 0.0% |

## Side effects

XSTest graded by an LLM judge (DeepSeek v4.1 flash, temperature 0). MMLU = Global-MMLU-Lite, 400 items per language. *Lenient* is a post-hoc parser added after seeing results (also accepts a letter alone on the first line), applied identically to every arm.

| Metric | A0 baseline | A1 prompt defence | A2 steering (CAA) | A4 DPO (SafeTune) |
|---|---|---|---|---|
| XSTest over-refusal (safe, ↓) | 8.8% | 11.2% | 8.4% | 8.4% |
| XSTest refusal (unsafe, ↑) | 96.5% | 99.0% | 96.0% | 97.0% |
| MMLU en strict (unparseable) | 60.0% (52) | 58.2% (52) | 61.0% (46) | 62.7% (39) |
| MMLU en lenient (unparseable) | 65.5% (16) | 61.8% (26) | 66.0% (10) | 65.2% (17) |
| MMLU hi strict (unparseable) | 47.0% (56) | 25.2% (206) | 43.2% (79) | 31.5% (171) |
| MMLU hi lenient (unparseable) | 49.2% (45) | 49.8% (30) | 50.5% (27) | 46.2% (69) |

## A2 steering grid (dev tuning subset: 50 injected, 50 clean, 50 authorized)

| Layer | α | ASR incl. relay | Clean success | Authorized followed |
|---|---|---|---|---|
| A0 | — | 32.0% | 44.0% | 16.0% |
| 14 | 4 | 26.0% | 42.0% | 16.0% |
| 14 **(chosen)** | 8 | 22.0% | 50.0% | 18.0% |
| 18 | 4 | 36.0% | 42.0% | 20.0% |
| 18 | 8 | 26.0% | 44.0% | 16.0% |
| 22 | 4 | 30.0% | 38.0% | 18.0% |
| 22 | 8 | 32.0% | 36.0% | 18.0% |
