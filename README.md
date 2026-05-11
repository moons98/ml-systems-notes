# ml-systems-notes

Reading notes on ML systems papers — inference engines, training systems, accelerators, edge AI, scheduling.

## Layout

- `notes/` — one file per paper. Filename: `<venue><year>-<short-title>.md`
- `concepts/` — reusable background notes referenced across papers (e.g., systolic arrays, UMA, W4A16). Extract a concept here when the same explanation would help in multiple papers.
- `queue.md` — backlog of papers to read.
- `pdfs/` — local PDF copies. Gitignored to keep the repo small.

## Index

See [INDEX.md](INDEX.md) for the chronological list of papers read.

## Note template

Each paper note follows this structure (top → bottom):

1. **Frontmatter** — venue, authors, tags, read date, status
2. **TL;DR** — one paragraph: problem + key insight + result
3. **Key Insights** — 3–5 transferable takeaways
4. **Background** — link out to `concepts/` for reusable ideas; only paper-specific quirks stay inline
5. **Walkthrough** — section-by-section, focused on *why* each design choice, not what
6. **Open Questions / Critique** — limitations, things to probe in a discussion
