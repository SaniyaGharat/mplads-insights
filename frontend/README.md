# MPLADS Insight

Build a dashboard called "MPLADS Fraud & Anomaly Detection" — a serious analytics tool for reviewing AI-flagged anomalies in MPLADS government fund sanctioning data (India). Audience: policy analysts / government auditors. It must look calm, professional, and trustworthy — a working tool, not a marketing site.

DESIGN DIRECTION (important):
- Color palette: sage green and navy blue as primary colors, soft neutral backgrounds (off-white / warm gray). Muted, editorial, calm — not vibrant or high-contrast.
- NO neon colors, no purple-blue gradients, no glowing effects, no glassmorphism, no generic AI-generated SaaS look (no indigo/violet gradient hero + emoji icons).
- Typography: editorial/institutional — serif or well-set sans-serif for headings, clean sans-serif for body/data.
- Data density over whitespace-heavy layouts. Subtle borders, muted status colors (not bright red/green) for badges/alerts.

COMPONENTS:
1. Risk-Ranked List (main view): data table with columns Rank, MP name, IDA (implementing agency) name, Amount (₹), Lag (days), Status, Score, and a Label Status badge with three visually distinct muted states: "Training Label", "GAT-Top", "New". Toggle to switch scoring method between "GAT" and "Siamese". Pagination (~15,000 rows, ~300 pages at 50/page). Clicking a row selects it and opens the explainability detail below or in a side panel.
2. Explainability Panel: shown on row selection; displays top 3 neighboring entities that influenced the row's anomaly score, each with a weight (0–1) shown as horizontal bars or a ranked list with the weight value visible.
3. Network Graph View: node-link graph, MPs and IDAs as differently shaped/colored nodes (circles for MPs, squares for IDAs), edges = sanctioned works; visualize ~top 50 highest-risk connections.
4. Labeling Widget: "Mark Suspicious" and "Mark Normal" buttons on each row or in the detail panel; running totals visible in header or sidebar of Suspicious / Normal / Skipped labels given.

BACKEND INTEGRATION — build against an existing FastAPI backend with these EXACT endpoints, field names, and response shapes (do not invent different names or shapes):
- GET /api/risk-scores?method=gat|siamese&page=1&page_size=50 → { "data": [ { work_index, MP, IDA, amount, sanction_lag_days, status, category, score, label_status } ... ], "page": 1, "page_size": 50, "total": 15000 }
- GET /api/graph?top_n=50 → { "nodes": [ { id, type: "MP"|"IDA" } ... ], "edges": [ { work_index, source, target, amount, lag, score } ... ] }
- GET /api/explain/{work_index} → { "work_index": int, "top_neighbors": [ { neighbor, weight } ... ] }
- POST /api/label with body { "work_index": int, "label": "s"|"n"|"skip" } → { "status": "success", "totals": { "s": int, "n": int, "skip": int } }

The base URL must be defined in ONE easily changeable place (e.g. a single constant/config), defaulting to http://localhost:8000/api for now; the user will swap in the real deployed URL later. Since the backend isn't reachable from the preview yet, handle failed requests gracefully (clear error/loading states), and it's fine to include a small set of realistic mock data used only as a fallback so the UI can be reviewed before the real URL is provided — clearly keep the real fetch logic wired to the exact endpoints. Keep it functional and clean over flashy.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/eaeac1b2-7ea4-4b5e-a69c-fd86c3b7aa77).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
