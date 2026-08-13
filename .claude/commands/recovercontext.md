---
description: Read handoff.md, verify it against actual repo state, and pick up work where the last session left off
---

# Resume

Read `handoff.md` and use it to reconstruct context from the last session, then verify that context is still accurate before continuing work.

## Instructions

### 1. Read the handoff

Read `handoff.md` from the project root. If it doesn't exist, tell the user and stop — don't guess at prior context.

### 2. Verify against actual state

Don't trust the file blindly — it may be stale. Check it against reality:

- **Git status/diff**: Compare the "Changes Made" and "Active Files" sections against `git status` and `git diff`. Note any discrepancies — files listed as changed that aren't, or uncommitted changes not mentioned in the handoff.
- **Current State claims**: If the handoff claims something like "tests passing," "build succeeds," or "feature X works," re-verify it — run the relevant test/build command. If the handoff doesn't specify what to run, infer from the project (package.json scripts, Makefile, CI config, etc.) or check briefly with the user if it's ambiguous.
- **Active Files**: Open and skim the listed files to confirm they still look like what the handoff describes, especially if git history shows changes since the handoff was written.

### 3. Read relevant codebase context

Beyond the files listed in "Active Files," pull in whatever additional context is needed to actually execute the "Next Steps" — related modules, tests, callers/callees of changed functions, relevant config. Don't just re-read what's listed; go one layer deeper if the next steps require it.

### 4. Report drift, if any

Before starting work, give a brief summary:
- What matches the handoff (confirm quickly, don't belabor it)
- What's drifted or is stale (be specific: "handoff says tests pass, but `npm test` now fails on X")
- Anything in "Failed Attempts" that's relevant to how you'll approach the next steps (don't repeat dead ends)

If drift is significant enough that "Next Steps" no longer makes sense (e.g. the goal appears already done, or the codebase has changed substantially), flag this clearly and ask the user how to proceed rather than pushing ahead on stale assumptions.

### 5. Begin the next step

Once state is confirmed (or discrepancies are surfaced and acknowledged), start on the first item in "Next Steps." State clearly which step you're starting so the user can redirect if priorities shifted.

## Output rules

- Keep the verification summary tight — a few lines, not a report. The user wants to get back to work, not read a status page.
- Don't re-explain the whole goal/history back to the user — they lived it. A one-line reorientation ("Picking up: fixing the null check in auth.ts") is enough.
- If `handoff.md` is missing or empty, stop and say so — don't fabricate context from git history alone.