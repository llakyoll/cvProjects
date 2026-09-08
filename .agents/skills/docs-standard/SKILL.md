---
name: docs-standard
description: Documentation standards for all cvProjects repositories. Use whenever writing or reviewing any README, docstring, comment, or documentation file in this repo, or when the user asks to document a project.
---

# Documentation Standard (cvProjects)

All documentation is written in **English**. Tone: clear, direct, professional.
No filler phrases ("this amazing project"), no emoji spam (max 1 emoji per
section header, none in body text).

## README structure — every project folder MUST have a README.md with exactly these sections, in this order:

1. **Title + one-line tagline** — what it does in one sentence.
2. **Demo** — a GIF or screenshot right under the title (`assets/demo.gif`).
   If no demo asset exists yet, insert the placeholder comment
   `<!-- TODO: demo.gif -->` and notify the user.
3. **Features** — 3-6 bullet points, each starting with a verb.
4. **How it works** — 2-5 sentences OR a Mermaid flowchart for anything with
   a pipeline (capture → process → output). Prefer the diagram.
5. **Installation** — numbered steps, tested commands, pinned requirements.
6. **Usage** — at least one copy-paste-ready command with example arguments,
   plus a table of CLI arguments (Argument | Description | Default) if the
   project has 3+ arguments.
7. **Project structure** — a short annotated tree (only if 4+ source files).
8. **Tech stack** — shields.io flat-style badges, not a text list.
9. **License + author footer** — link to GitHub profile and LinkedIn.

## Code documentation rules

- Every module: a one-paragraph module docstring (what + why).
- Every public function/class: Google-style docstring (Args/Returns/Raises).
- Comments explain *why*, never *what* the code obviously does.
- Config values never hardcoded in logic — pulled to top-level CONFIG or CLI args.

## Naming conventions

- Folders: `kebab-case` (e.g., `vehicle-counting`)
- Python files: `snake_case.py`
- Every project folder contains: `README.md`, `requirements.txt`, `main.py`,
  `src/`, `assets/`

## Quality checklist before finishing any documentation task

- [ ] Would a stranger be able to run this project in under 5 minutes?
- [ ] Is there a visual (GIF/diagram) above the fold?
- [ ] Are all commands actually runnable (no pseudo-commands)?
- [ ] Zero Turkish text in committed files?
