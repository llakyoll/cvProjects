---
name: readme-generator
description: Generate or update a README.md for a project folder following the cvProjects documentation standard. Use when the user asks to create, write, improve, or regenerate a README for any project in this repo.
---

# README Generator

Generates READMEs that follow the `docs-standard` skill. Always load and
follow `docs-standard` when this skill runs.

## Process

1. **Analyze the project folder first** — read `main.py`, everything under
   `src/`, and `requirements.txt`. Extract: purpose, pipeline stages, CLI
   arguments (argparse definitions), key dependencies, entry point.
   Never invent features that aren't in the code.
2. **Draft the README** using the exact section order from `docs-standard`.
3. **Visuals:** if `assets/demo.gif` exists, embed it. If not, add the
   placeholder and tell the user which command from the `visual-assets`
   skill would generate it. Add a Mermaid pipeline diagram whenever the
   code has 2+ processing stages.
4. **Update the root index:** after writing a project README, update the
   project table in the repository root `README.md` (name, one-line
   description, stack, demo thumbnail) so the index never goes stale.
5. **Verify:** run the docs-standard quality checklist. Report any
   unchecked items to the user instead of silently passing.

## Style constraints

- Tagline: max 15 words, no buzzwords ("cutting-edge", "state-of-the-art").
- Usage examples must use realistic arguments
  (`--source rtsp://192.168.1.64/stream1`, not `--source <your-source>`
  as the only example).
- Badges: only technologies actually used in that project.
- Footer template:
  `**Ahmed Akyol** — Computer Vision Engineer · [GitHub](https://github.com/llakyoll) · [LinkedIn](https://www.linkedin.com/in/ahmed-akyol-84766622b/)`
