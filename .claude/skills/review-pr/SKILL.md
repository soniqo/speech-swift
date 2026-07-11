---
name: review-pr
description: Review a pull request for conceptual fit, architecture impact, adversarial failure modes, security risk, docs impact, regression risk, test coverage, and merge readiness. Use when asked to review a PR, check whether a PR is safe to merge, decide if more tests are needed, perform adversarial or security review, or summarize PR risk.
disable-model-invocation: false
argument-hint: <PR number or URL>
allowed-tools: Bash
---

# Review PR

Review the PR from the user's point of view: make the decision easy, then provide detail only when it changes the decision.

## Workflow

1. Inspect PR metadata, files, diff, and checks.
2. Read the touched code and nearby architecture boundaries.
3. Identify whether the change is a behavior change, performance change, refactor, test-only change, docs-only change, or CI/infrastructure change.
4. Run the required review passes:
   - architecture fit: does the change match existing module boundaries and project patterns;
   - adversarial review: what breaks under edge cases, bad inputs, scale, concurrency, cancellation, cache misses, offline mode, or hosted runner differences;
   - security review: secrets, tokens, path traversal, unsafe downloads, sandbox escapes, user-controlled file paths, network exposure, dependency risk, and data leakage;
   - docs review: local docs, public site docs, README translations, CLI help, model tables, benchmark pages, and migration notes when behavior is user-visible.
5. Assess regression risk from the changed behavior and blast radius.
6. Decide whether existing tests are enough. Include E2E coverage for runtime/model/user-facing behavior, or explicitly state why E2E is not applicable. Add or request tests only when they protect a real risk.
7. Run the smallest meaningful validation:
   - unit tests for logic and data structures;
   - focused E2E for model/runtime behavior;
   - CLI probe or benchmark only when the PR changes user-facing runtime behavior or performance.
8. End with a short merge recommendation.

## Output Shape

Start with this format unless the user asks for a detailed code review:

```md
Status: ready / not ready / needs follow-up

What changed:
...

Architecture fit:
...

Adversarial/security/docs:
Adversarial: ...
Security: ...
Docs: ...

Regression risk:
Low / medium / high, with one-sentence reason.

Tests:
Unit: ...
E2E: ...
Needed: ...

Recommendation:
Merge / request changes / investigate first.
```

## Review Rules

- Lead with blockers if any exist.
- Keep conceptual summary above command details.
- Do not paste logs unless asked.
- Be adversarial about failure modes, but do not invent blockers without evidence.
- Treat security and docs as explicit review passes, even when the conclusion is "no impact."
- Always report E2E tests run. If none were run, state the reason and whether that is acceptable.
- Do not add slow permanent E2E tests just to prove a local probe; add permanent tests when they guard realistic future regressions.
- Prefer same-input comparison against `main` for performance or refactor PRs when output preservation matters.
- Treat model download, cache, and hosted runner failures as infrastructure unless code behavior caused them.
- For public API, CLI flags, model variants, or docs-visible behavior, call out required docs updates.
- For README changes, verify all translated README files are updated.
