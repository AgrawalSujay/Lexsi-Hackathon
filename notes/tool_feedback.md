# AuditKIT integration feedback

Version inspected: **AuditKIT 1.1.6** (installed Python package).
Context: importing recorded OpenAI-format agent traces and checking verified
outcomes with `FinalStateAssertion`; no agent or judge calls.

| Friction point | Observed behavior | Suggestion |
|---|---|---|
| Required task | `AgentCase(id=..., outcome=...)` is insufficient: `task` must also be nonempty, even in recorded mode. | Document a minimal recorded-mode example including `task`, or infer it from the imported user request. |
| Final-state coverage | `episode_from_openai_messages` sets final-state coverage to unavailable; assigning `episode.final_state` later does not update it. | Provide an `attach_final_state` method that updates the snapshot and coverage together. |
| Missing key versus `None` | A missing assertion key produces `unknown`; a present `None` compared with `equals=False` produces failure. | Document this distinction and offer an explicit missing-value policy for nullable imported evidence. |
| Errors override assertions | `stop_reason="error"` short-circuits assertions, even when the partial trace proves an unsafe action already happened. | Allow verified effects to remain scoreable on errored partial runs while preserving error provenance. |
| Example not packaged | The task references `examples/agent_eval_offline.py`, but it is absent from the installed distribution. | Include the example in the wheel or link directly to it from the recorded-mode API documentation. |

TurnGuard currently supplies the recorded user request, updates coverage when
attaching observed state, omits unavailable assertion keys, and retains errors
on known unsafe partial runs without setting the overriding `error` stop reason.
Safety reports contain graded injected episodes only; task reports contain all
graded episodes. Ungraded runs remain visible in TurnGuard's core run artifacts.
