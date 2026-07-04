---
name: review-pr
description: Review a pull request for conceptual fit, architecture impact, regression risk, test coverage, and merge readiness. Use when asked to review a PR, check whether a PR is safe to merge, decide if more tests are needed, or summarize PR risk.
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
4. Assess regression risk from the changed behavior and blast radius.
5. Decide whether existing tests are enough. Add or request tests only when they protect a real risk.
6. Run the smallest meaningful validation:
   - unit tests for logic and data structures;
   - focused E2E for model/runtime behavior;
   - CLI probe or benchmark only when the PR changes user-facing runtime behavior or performance.
7. End with a short merge recommendation.

## Output Shape

Start with this format unless the user asks for a detailed code review:

```md
Status: ready / not ready / needs follow-up

What changed:
...

Architecture fit:
...

Regression risk:
Low / medium / high, with one-sentence reason.

Tests:
Ran: ...
Needed: ...

Recommendation:
Merge / request changes / investigate first.
```

## Review Rules

- Lead with blockers if any exist.
- Keep conceptual summary above command details.
- Do not paste logs unless asked.
- Do not add slow permanent E2E tests just to prove a local probe; add permanent tests when they guard realistic future regressions.
- Prefer same-input comparison against `main` for performance or refactor PRs when output preservation matters.
- Treat model download, cache, and hosted runner failures as infrastructure unless code behavior caused them.
- For public API, CLI flags, model variants, or docs-visible behavior, call out required docs updates.
- For README changes, verify all translated README files are updated.
