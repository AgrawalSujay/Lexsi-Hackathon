# External detector baseline: `meta-llama/Llama-Prompt-Guard-2-86M`

Positive label(s): label_1. Threshold 0.5. Episode flagged if any injected/controlled slot scores ≥ threshold (512-token windows, max).

## test

| Metric | Rate |
|---|---|
| Injections detected (whole tool field) | 0.0% (0/407) |
| Injections detected (injected sentence alone) | 0.0% (0/407) |
| False alarms on clean twins | 0.0% (0/407) |
| False blocks on authorized instructions | 0.0% (0/229) |

Detection by lang: en 0.0% (0/145); hi 0.0% (0/135); hinglish 0.0% (0/127)

Detection by style: fake_authority 0.0% (0/68); imperative 0.0% (0/68); named_approval 0.0% (0/70); protocol_spoof 0.0% (0/59); quiet_fyi 0.0% (0/72); task_framed 0.0% (0/70)

Detection by goal: destroy 0.0% (0/81); exfiltrate 0.0% (0/86); extra_call 0.0% (0/78); redirect 0.0% (0/78); tamper_answer 0.0% (0/84)

Detection by where: state 0.0% (0/309); user_turn 0.0% (0/98)

False alarms (clean) by conversation language: en 0.0% (0/143); hi 0.0% (0/133); hinglish 0.0% (0/131)

False blocks (authorized) by lang: en 0.0% (0/72); hi 0.0% (0/80); hinglish 0.0% (0/77)

| Threshold | Detect (field) | Detect (snippet) | False alarm (clean) | False block (authorized) |
|---|---|---|---|---|
| 0.1 | 0.2% | 0.0% | 0.0% | 0.0% |
| 0.2 | 0.2% | 0.0% | 0.0% | 0.0% |
| 0.3 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.4 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.5 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.6 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.7 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.8 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.9 | 0.0% | 0.0% | 0.0% | 0.0% |

## test_cueflip

| Metric | Rate |
|---|---|
| Injections detected (whole tool field) | 0.0% (0/107) |
| Injections detected (injected sentence alone) | 0.0% (0/107) |
| False alarms on clean twins | 0.0% (0/107) |
| False blocks on authorized instructions | 0.0% (0/56) |

Detection by lang: en 0.0% (0/39); hi 0.0% (0/32); hinglish 0.0% (0/36)

Detection by style: fake_authority 0.0% (0/20); imperative 0.0% (0/19); named_approval 0.0% (0/16); protocol_spoof 0.0% (0/15); quiet_fyi 0.0% (0/20); task_framed 0.0% (0/17)

Detection by goal: destroy 0.0% (0/19); exfiltrate 0.0% (0/22); extra_call 0.0% (0/22); redirect 0.0% (0/22); tamper_answer 0.0% (0/22)

Detection by where: state 0.0% (0/79); user_turn 0.0% (0/28)

False alarms (clean) by conversation language: en 0.0% (0/39); hi 0.0% (0/32); hinglish 0.0% (0/36)

False blocks (authorized) by lang: en 0.0% (0/20); hi 0.0% (0/19); hinglish 0.0% (0/17)

| Threshold | Detect (field) | Detect (snippet) | False alarm (clean) | False block (authorized) |
|---|---|---|---|---|
| 0.1 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.2 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.3 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.4 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.5 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.6 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.7 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.8 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.9 | 0.0% | 0.0% | 0.0% | 0.0% |
