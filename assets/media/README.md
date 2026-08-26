# Submission media

The demo video and its README preview, embedded in the
top-level README and described in [`../../docs/design-document.md`](../../docs/design-document.md) §7.

Committed deliberately: `*.mp4` is gitignored so that generated captures under `outputs/` and
`runs/` stay out of the repo, and this directory is the explicit exception. Keep files small
enough to serve inline on GitHub — re-encode rather than committing a large capture.

| File | Content | Status |
|---|---|---|
| `demo.mp4` | 69 s, four segments: live stereo-hand free play (two unedited takes, one of which fails), four takes from the blinded human-operator trial, the analytical expert against no assist, and what the measurements support | shipped |
| `demo-preview.gif` | 6.5 s silent excerpt of segment 1, embedded in the top-level README because GitHub strips `<video>` tags from rendered Markdown | shipped |
