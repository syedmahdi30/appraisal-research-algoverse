# Handoff

Generate a `handoff.md` file that captures the current session's state so a fresh Claude Code session can pick up immediately without re-discovering context.

## When to use

Run this before ending a session that has unfinished work, or any time you want a checkpoint before compacting/clearing context.

## Instructions

Create a file named `handoff.md` in the project root (overwrite if it already exists). Populate it using the six sections below, in this exact order. Base the content on the actual conversation and repo state — inspect git status/diff and recently edited files rather than guessing.

### 1. Goal
One or two sentences: what is the user actually trying to accomplish in this session? State the end objective, not the immediate sub-task.

### 2. Current State
Where things stand right now. Is the build passing? Are tests green? Is the feature half-implemented? Be concrete and honest — this is not a status report to impress anyone, it's a snapshot for a cold start.

### 3. Active Files
List the files currently being worked on or most relevant to the goal, with a short note on each (what it does / why it matters right now). Use file paths relative to the repo root. Pull this from git status/diff plus files opened or edited during the session.

### 4. Changes Made
A concise, chronological-ish list of what was actually changed this session — code edits, config changes, new files, dependencies added. Summarize diffs rather than pasting full diffs unless a snippet is essential to understanding.

### 5. Failed Attempts
Anything that was tried and didn't work, and — critically — *why* it didn't work, if known. This is the highest-value section: it prevents the next session from re-treading dead ends. Include approaches considered and rejected, not just runtime errors.

### 6. Next Steps
A concrete, ordered list of what to do next. Prefer imperative, actionable items ("Fix the null check in auth.ts:42", not "look into auth issues"). If there's a clear next single step, put it first.

## Output rules

- Write real content, not placeholders. If a section genuinely has nothing (e.g., no failed attempts yet), write "None yet" rather than omitting the section.
- Keep it dense and skimmable — this file should be readable in under a minute.
- Use `##` headers for each of the six sections, in order, exactly as named above.
- Do not include unrelated boilerplate (no "Introduction," no "Summary" at the end).