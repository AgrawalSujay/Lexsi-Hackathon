# External detector baseline: `protectai/deberta-v3-base-prompt-injection-v2`

Positive label(s): injection. Threshold 0.5. Episode flagged if any injected/controlled slot scores ≥ threshold (512-token windows, max).

## test

| Metric | Rate |
|---|---|
| Injections detected (whole tool field) | 60.0% (244/407) |
| Injections detected (injected sentence alone) | 76.2% (310/407) |
| False alarms on clean twins | 37.1% (151/407) |
| False blocks on authorized instructions | 48.0% (110/229) |

Detection by lang: en 44.1% (64/145); hi 73.3% (99/135); hinglish 63.8% (81/127)

Detection by style: fake_authority 47.1% (32/68); imperative 61.8% (42/68); named_approval 57.1% (40/70); protocol_spoof 66.1% (39/59); quiet_fyi 69.4% (50/72); task_framed 58.6% (41/70)

Detection by goal: destroy 61.7% (50/81); exfiltrate 52.3% (45/86); extra_call 52.6% (41/78); redirect 53.8% (42/78); tamper_answer 78.6% (66/84)

Detection by where: state 63.1% (195/309); user_turn 50.0% (49/98)

False alarms (clean) by conversation language: en 12.6% (18/143); hi 60.9% (81/133); hinglish 39.7% (52/131)

False blocks (authorized) by lang: en 52.8% (38/72); hi 45.0% (36/80); hinglish 46.8% (36/77)

| Threshold | Detect (field) | Detect (snippet) | False alarm (clean) | False block (authorized) |
|---|---|---|---|---|
| 0.1 | 68.3% | 81.3% | 46.9% | 53.3% |
| 0.2 | 65.8% | 78.9% | 45.5% | 51.5% |
| 0.3 | 63.4% | 77.9% | 44.2% | 50.7% |
| 0.4 | 60.9% | 76.7% | 37.6% | 48.9% |
| 0.5 | 60.0% | 76.2% | 37.1% | 48.0% |
| 0.6 | 59.0% | 75.4% | 37.1% | 47.2% |
| 0.7 | 59.0% | 74.7% | 36.9% | 47.2% |
| 0.8 | 57.2% | 73.5% | 36.1% | 45.9% |
| 0.9 | 54.0% | 72.0% | 35.6% | 43.2% |

## test_cueflip

| Metric | Rate |
|---|---|
| Injections detected (whole tool field) | 41.1% (44/107) |
| Injections detected (injected sentence alone) | 63.5% (68/107) |
| False alarms on clean twins | 24.3% (26/107) |
| False blocks on authorized instructions | 53.6% (30/56) |

Detection by lang: en 12.8% (5/39); hi 78.1% (25/32); hinglish 38.9% (14/36)

Detection by style: fake_authority 25.0% (5/20); imperative 52.6% (10/19); named_approval 50.0% (8/16); protocol_spoof 53.3% (8/15); quiet_fyi 50.0% (10/20); task_framed 17.6% (3/17)

Detection by goal: destroy 26.3% (5/19); exfiltrate 31.8% (7/22); extra_call 40.9% (9/22); redirect 50.0% (11/22); tamper_answer 54.5% (12/22)

Detection by where: state 38.0% (30/79); user_turn 50.0% (14/28)

False alarms (clean) by conversation language: en 10.3% (4/39); hi 43.8% (14/32); hinglish 22.2% (8/36)

False blocks (authorized) by lang: en 75.0% (15/20); hi 36.8% (7/19); hinglish 47.1% (8/17)

| Threshold | Detect (field) | Detect (snippet) | False alarm (clean) | False block (authorized) |
|---|---|---|---|---|
| 0.1 | 49.5% | 70.1% | 34.6% | 60.7% |
| 0.2 | 47.7% | 67.3% | 33.6% | 58.9% |
| 0.3 | 45.8% | 65.4% | 32.7% | 57.1% |
| 0.4 | 43.9% | 63.5% | 25.2% | 55.4% |
| 0.5 | 41.1% | 63.5% | 24.3% | 53.6% |
| 0.6 | 41.1% | 63.5% | 24.3% | 53.6% |
| 0.7 | 40.2% | 60.8% | 24.3% | 53.6% |
| 0.8 | 39.2% | 58.9% | 24.3% | 53.6% |
| 0.9 | 36.4% | 56.1% | 24.3% | 53.6% |
