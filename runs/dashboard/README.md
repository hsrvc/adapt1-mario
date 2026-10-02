# Frozen runs with per-move scores (2026-10-02)

One frozen 1-1 run per pair, recorded by `scripts/dashboard_live.py`: exploration off, no feedback, both models
confirmed installed first. Every decision is saved with the server's response, including `values`, the expected
reward for each move. These are the scores the dashboard clips show.

| file | pair | result |
|---|---|---|
| `zero-start-2026-10-02.json` | the zero-start learners | 3161 (flag), 47 decisions, all chosen by Adapt-1 |
| `warm-start-2026-10-02.json` | the warm-start learners | 3161 (flag), 62 decisions, all chosen by Adapt-1 |

Re-render a clip offline (no API calls): `python scripts/dashboard_live.py --from-evidence <file> --out clip.mp4`. Recording a new run needs a key and costs about 80 Queries, 0 Records.
