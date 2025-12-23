# Repository Guidelines

## Project Structure & Module Organization
- This repo is an Obsidian vault of Markdown notes; core content lives under folders such as `学习/`, `科研/`, `实验/`, and `杂项/` for study notes, research, experiments, and misc configs.
- Visual assets from Excalidraw reside in `Excalidraw/`; keep diagram files there to retain plugin links.
- Vault configuration is in `.obsidian/`; avoid manual edits unless you are adjusting verified plugin or theme settings.
- Utility scripts live in `tools/` (e.g., `tools/pptx_to_pdf.py` for converting PPTX to PDF).

## Build, Test, and Development Commands
- Notes require no build; open the vault directly in Obsidian.
- Convert slides: `python tools/pptx_to_pdf.py` (uses current directory) or `python tools/pptx_to_pdf.py <folder>` to emit PDFs alongside PPTX files. Requires Windows PowerPoint and `comtypes`.
- Git hygiene: `git status` before and after changes; use `git add -A` to capture new/renamed notes.

## Coding Style & Naming Conventions
- Notes: Markdown only, clear H1 titles, short sections, and Obsidian-friendly internal links. Keep language consistent with the existing note’s language (Chinese or English).
- File placement: choose the closest existing folder; create new folders sparingly with descriptive names.
- Python (in `tools/`): follow PEP 8 (4-space indent, snake_case). Keep docstrings bilingual only if the module already mixes languages.

## Testing Guidelines
- No automated test suite. For script changes, run `python tools/pptx_to_pdf.py <folder_with_sample_pptx>` and confirm PDFs are produced without console errors.
- If adding new scripts, include a minimal usage example in a top-level docstring and validate on a small sample input.

## Commit & Pull Request Guidelines
- Commit history uses `vault backup: YYYY-MM-DD HH:MM:SS`; match this for routine snapshot commits. Use descriptive prefixes (e.g., `feat:`, `fix:`) only when making targeted feature/script changes.
- Commits should stay focused: related note edits per topic, or a single script change plus its usage note.
- PRs (if used) should list scope, key directories touched, and any manual verification steps (e.g., script run results or screenshots of rendered notes). Link related issues/tasks when applicable.

## Security & Configuration Tips
- Do not commit private tokens, certificates, or account data in notes or `.obsidian/`.
- Back up before large refactors of folder structure; Obsidian links rely on stable paths.
- Keep binary assets (PDF/PPTX) small and only when needed; prefer linking to external storage if files are large.
