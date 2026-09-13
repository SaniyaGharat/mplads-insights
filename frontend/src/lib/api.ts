import { API_BASE_URL } from "./config";

export type ScoringMethod = "gat" | "siamese";
export type LabelStatus = "Training Label" | "GAT-Top" | "New" | string;

export interface RiskRow {
  work_index: number;
  MP: string;
  IDA: string;
  amount: number;
  sanction_lag_days: number;
  status: string;
  category: string;
  score: number;
  label_status: LabelStatus;
}

export interface RiskScoresResponse {
  data: RiskRow[];
  page: number;
  page_size: number;
  total: number;
}

export interface GraphNode {
  id: string;
  type: "MP" | "IDA";
}

export interface GraphEdge {
  work_index: number;
  source: string;
  target: string;
  amount: number;
  lag: number;
  score: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ExplainResponse {
  work_index: number;
  top_neighbors: { neighbor: string; weight: number }[];
}

export type LabelValue = "s" | "n" | "skip";

export interface LabelTotals {
  s: number;
  n: number;
  skip: number;
}

export interface LabelResponse {
  status: string;
  totals: LabelTotals;
}

/** Result wrapper so the UI can tell live data from mock fallback. */
export interface Sourced<T> {
  data: T;
  source: "live" | "mock";
  error?: string;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

// ---------------- real endpoints (exact shapes) ----------------

export async function fetchRiskScores(
  method: ScoringMethod,
  page: number,
  pageSize: number,
): Promise<Sourced<RiskScoresResponse>> {
  try {
    const data = await getJson<RiskScoresResponse>(
      `/risk-scores?method=${method}&page=${page}&page_size=${pageSize}`,
    );
    return { data, source: "live" };
  } catch (e) {
    return {
      data: mockRiskScores(method, page, pageSize),
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchGraph(topN = 50): Promise<Sourced<GraphResponse>> {
  try {
    const data = await getJson<GraphResponse>(`/graph?top_n=${topN}`);
    return { data, source: "live" };
  } catch (e) {
    return { data: mockGraph(topN), source: "mock", error: (e as Error).message };
  }
}

export async function fetchExplain(workIndex: number): Promise<Sourced<ExplainResponse>> {
  try {
    const data = await getJson<ExplainResponse>(`/explain/${workIndex}`);
    return { data, source: "live" };
  } catch (e) {
    return { data: mockExplain(workIndex), source: "mock", error: (e as Error).message };
  }
}

export async function postLabel(
  workIndex: number,
  label: LabelValue,
): Promise<Sourced<LabelResponse>> {
  try {
    const res = await fetch(`${API_BASE_URL}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ work_index: workIndex, label }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return { data: (await res.json()) as LabelResponse, source: "live" };
  } catch (e) {
    mockTotals[label] += 1;
    return {
      data: { status: "success", totals: { ...mockTotals } },
      source: "mock",
      error: (e as Error).message,
    };
  }
}

// ---------------- deterministic mock fallback ----------------

export const MOCK_TOTAL = 15000;

const MPS = [
  "Dr. A. Ramanathan",
  "Smt. Kavita Deshmukh",
  "Shri P. K. Bansal",
  "Shri Rakesh Mahto",
  "Smt. Nandini Rao",
  "Shri T. Selvaraj",
  "Shri Imran Qureshi",
  "Smt. Meera Joshi",
  "Shri D. N. Patil",
  "Shri Harbhajan Gill",
];

const IDAS = [
  "PWD Division, Nashik",
  "Zilla Parishad, Guntur",
  "Municipal Corp., Kanpur",
  "Rural Works Dept., Ranchi",
  "Water Board, Coimbatore",
  "District Collectorate, Patna",
  "Housing Board, Indore",
  "Panchayat Samiti, Bardhaman",
];

const STATUSES = ["Sanctioned", "In Progress", "Completed", "Withheld"];
const CATEGORIES = ["Roads & Bridges", "Sanitation", "Education", "Health", "Water Supply"];
const LABELS: LabelStatus[] = ["Training Label", "GAT-Top", "New"];

const mockTotals: LabelTotals = { s: 0, n: 0, skip: 0 };

function at<T>(arr: T[], i: number): T {
  return arr[((i % arr.length) + arr.length) % arr.length] as T;
}

function seeded(n: number) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function mockRiskScores(
  method: ScoringMethod,
  page: number,
  pageSize: number,
): RiskScoresResponse {
  const offset = (page - 1) * pageSize;
  const bias = method === "gat" ? 0 : 7;
  const data: RiskRow[] = Array.from({ length: pageSize }, (_, i) => {
    const rank = offset + i + 1;
    const r = seeded(rank + bias);
    return {
      work_index: 100000 + rank * 7 + bias,
      MP: at(MPS, rank + bias),
      IDA: at(IDAS, rank * 3 + bias),
      amount: Math.round((5 + r * 240) * 100000),
      sanction_lag_days: Math.round(4 + seeded(rank * 2 + bias) * 730),
      status: at(STATUSES, rank + bias),
      category: at(CATEGORIES, rank * 2 + bias),
      score: Math.max(0.01, 0.995 - rank * 0.00006 - r * 0.02),
      label_status: at(LABELS, rank + bias),
    };
  });
  return { data, page, page_size: pageSize, total: MOCK_TOTAL };
}

function mockGraph(topN: number): GraphResponse {
  const rows = mockRiskScores("gat", 1, topN).data;
  const nodes: GraphNode[] = [];
  const seen = new Set<string>();
  const push = (id: string, type: "MP" | "IDA") => {
    if (seen.has(id)) return;
    seen.add(id);
    nodes.push({ id, type });
  };
  const edges: GraphEdge[] = rows.map((row) => {
    push(row.MP, "MP");
    push(row.IDA, "IDA");
    return {
      work_index: row.work_index,
      source: row.MP,
      target: row.IDA,
      amount: row.amount,
      lag: row.sanction_lag_days,
      score: row.score,
    };
  });
  return { nodes, edges };
}

function mockExplain(workIndex: number): ExplainResponse {
  const base = seeded(workIndex);
  const pick = (offset: number) => {
    const v = seeded(workIndex * (offset + 3) + offset * 17);
    return v > 0.5
      ? at(IDAS, Math.floor(v * IDAS.length))
      : at(MPS, Math.floor(v * MPS.length));
  };
  const weights = [0.62 + base * 0.3, 0.35 + base * 0.25, 0.12 + base * 0.2];
  return {
    work_index: workIndex,
    top_neighbors: [0, 1, 2].map((i) => ({
      neighbor: pick(i + 1),
      weight: Math.min(0.99, Number(at(weights, i).toFixed(2))),
    })),
  };
}

export function formatINR(amount: number): string {
  return `₹${amount.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

// ---------------- stats ----------------

export interface StatsResponse {
  total_works: number;
  total_amount: number;
  avg_lag_days: number;
  label_counts: { training_label: number; gat_top: number; new: number };
  status_breakdown: { status: string; count: number }[];
  score_distribution: { bucket: string; count: number }[];
  top_risky_mps: { MP: string; flagged_count: number; total_flagged_amount: number }[];
  avg_lag_flagged_vs_normal: { flagged_avg_lag: number; normal_avg_lag: number };
}

export async function fetchStats(): Promise<Sourced<StatsResponse>> {
  try {
    const data = await getJson<StatsResponse>(`/stats`);
    return { data, source: "live" };
  } catch (e) {
    return { data: mockStats(), source: "mock", error: (e as Error).message };
  }
}

function mockStats(): StatsResponse {
  const status_breakdown = STATUSES.map((status, i) => ({
    status,
    count: Math.round(1200 + seeded(i + 1) * 4200),
  }));
  // Equal-count (percentile) buckets: ~10% of works per bucket, so bar
  // heights stay roughly level. The skew shows up as bucket WIDTH — the
  // final bucket spans a huge score range while the low-score buckets are
  // narrow.
  const percentileBuckets = [
    "0.0–0.02",
    "0.02–0.05",
    "0.05–0.10",
    "0.10–0.18",
    "0.18–0.31",
    "0.31–0.55",
    "0.55–1.10",
    "1.10–2.40",
    "2.40–6.80",
    "6.80–13913.7",
  ];
  const score_distribution = percentileBuckets.map((bucket, i) => ({
    bucket,
    count: Math.round(1480 + seeded(i + 5) * 160 - i * 8),
  }));
  const top_risky_mps = MPS.map((MP, i) => ({
    MP,
    flagged_count: Math.round(38 + seeded(i + 11) * 120),
    total_flagged_amount: Math.round((40 + seeded(i + 21) * 260) * 100000),
  })).sort((a, b) => b.flagged_count - a.flagged_count);
  return {
    total_works: MOCK_TOTAL,
    total_amount: 8_642_000_000,
    avg_lag_days: 187,
    label_counts: { training_label: 1240, gat_top: 500, new: 13260 },
    status_breakdown,
    score_distribution,
    top_risky_mps,
    avg_lag_flagged_vs_normal: { flagged_avg_lag: 412, normal_avg_lag: 143 },
  };
}
