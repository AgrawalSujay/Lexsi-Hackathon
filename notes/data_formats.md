# Data formats — scenario templates and injection bank (task 03, Step 2)

Two YAML inputs feed the builder:

- `turnguard/data/templates/<family>.yaml`: **scenarios**, the user's task, the
  tool state, where injections can go, and how success is checked.
- `turnguard/data/injections.yaml`: the **injection bank**, what the attacker
  wants (goal), how it is framed (style), pressure, split forms, and attacker
  resources per split.

Filled examples: `email.yaml` (`email-reschedule`), `tickets.yaml`
(`tickets-where-is-order`). `scripts/render_prototype.py` renders every
combination of them with stub filler and checks each episode through the real
harness (866/866 pass). It is the reference for Step 3's assembly logic.

## 1. Scenario (`templates/<family>.yaml`)

| Key | Meaning |
|---|---|
| `id` | Unique scenario ID; becomes the episode ID prefix. |
| `tools` | Tools offered in the episode (must exist for the family). |
| `goals` | Which bank goals make sense here. |
| `vars` | Fixed values substituted as `{name}` everywhere. **Language-neutral**: IDs, codes, addresses, numbers. |
| `assets` | Values the injection bank uses: `sensitive_item`, `task_ref`, `approver`, `system_name`, `destroy_verb`, `destroy_ref`, `extra_verb`. |
| `turns` | One English string per user turn. For hi/hinglish conversations the teacher translates them, keeping every `{var}` value verbatim. |
| `state` | Initial tool state. A value `{gen: kind, about, words, include}` is written by the teacher in the conversation language; every `include` string must appear verbatim exactly once. Other values are literal. |
| `slots` | Where an injection goes, per position (`first` / `later` / `split`) and carrier (`body` / `metadata` / `quoted` / `tool_error`). See §3. |
| `route` | One legitimate solution: `{turn, tool, args}` calls per user turn. Used only for checks. |
| `obey_args` | Checks only: arguments an obeying agent would use where the target leaves them open (e.g. which ticket it replies on). |
| `task_oracle` | Matchers (task 02 schema). **Terms must be language-neutral**: each contains a digit, `@` or `/`. |
| `authorized` | Variants where the user delegates: `position`, `carrier`, `delegate` (appended to that user turn), `text` (the planted, now legitimate instruction), `target` matchers. |

## 2. Injection bank (`injections.yaml`)

| Key | Meaning |
|---|---|
| `attacker_pools.<split>` | `addr` / `url` / `user` values, disjoint across train, dev and test, all on `*.invalid`. |
| `goals.<goal>` | `ask` (the request, a clause) and `targets` per family (`"*"` = any family). Call targets carry a distinctive `args_contains` term (attacker address / URL / user), so the relay diagnostic can check them. Exception: `destroy`, whose target is a decoy ID (`args_match`). |
| `styles.<style>` | `long` (standalone texts, one sampled) and `short` (a clause for snag and split texts). Styles: `imperative`, `fake_authority`, `named_approval`, `task_framed`. |
| `pressure.snag` | `tool_error`: `{error, inject}` pairs. `content`: obstacle texts for body / metadata / quoted. |
| `split` | Pairs of texts: part 1 primes, part 2 carries the ask. |

Text is built in this order: `ask` ← goal and assets; `short` ← style;
injection text ← one of:
- `long` (pressure none);
- `pressure.snag.content` (snag on body / metadata / quoted);
- `pressure.snag.tool_error` (carrier tool_error, always);
- a `split` pair (position split).

## 3. Placement rules

- **`at: <JSON Pointer>`** appends `"\n\n" + text` to that string in the state.
  The schema part is `{where: state, slot: <pointer>, text}`.
- **`turn: i, intro`** appends `\n\n<intro>\n"<text>"` to user turn `i`
  (0-based): the user pasted content that contained it (carrier `quoted`).
- **`tool: T, on_call: n`** adds a scripted error to `tool_errors[T]`. The
  output is `error + " " + inject`, and the injection part is only `inject`, so
  the clean twin keeps the plain error. The route retries the call.
- **Main path.** Each slot is revealed by the **first route call of the user
  turn it belongs to**. `later` = a later user turn, still on that turn's main
  path. `split` = two slots, both on the route. Requests name the IDs they need,
  so no ID guessing is required.
- **Clean twin** = `make_clean_twin(injected)`, unchanged from task 02.

## 4. Languages

`lang.conversation` (turns + generated filler) and `lang.injection` are sampled
independently from en / hi / hinglish, which covers language mismatch.

- Generated filler is written directly in the conversation language.
- Turns and the final injection text are translated by the teacher.
- **`keep` tokens** (every substituted `{var}`, asset and attacker value) must
  survive translation verbatim and exactly once. The builder rejects any
  translation that drops, alters or duplicates one.
- Hand-written `hi:` / `hinglish:` overrides may sit next to `long` and are used
  instead of the teacher.

## 5. Checks the builder runs on every episode (prototype already does 1–4)

1. `validate_episode` passes (twins included).
2. Route alone: injected → `safe`, clean → `pass`, authorized → `pass`.
3. Route: every injection / authorized part is exposed (`oracles.exposed`).
4. Route + obeying (using `obey_args`) → `unsafe`.
5. Oracle terms are language-neutral; `include` strings appear exactly once.
6. Injection text appears only at its slots; the clean twin contains none of it.
7. Generated text contains no emails or URLs outside `*.invalid` / `example.*`.

## 6. Factor coverage per scenario

Carriers × positions come from `slots`. A scenario that lacks a slot simply
doesn't produce that combination, and the sampler balances across scenarios.
Held-out rules (`code` family, `tool_error` carrier → test only) are applied by
the sampler, not the templates.
