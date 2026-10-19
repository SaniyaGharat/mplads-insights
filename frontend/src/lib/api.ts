import { API_BASE_URL } from "./config";

export type ScoringMethod = "gat" | "siamese" | "community" | "fusion";
export type LabelStatus = "Training Label" | "GAT-Top" | "New" | string;

export interface MethodInfo {
  id: ScoringMethod;
  name: string;
  direction: "higher_is_riskier" | "lower_is_riskier";
  note: string;
}

export interface MethodsResponse {
  methods: MethodInfo[];
}

export interface RiskRow {
  work_index: number;
  global_rank?: number;
  MP: string;
  IDA: string;
  amount: number;
  sanction_lag_days: number;
  status: string;
  category: string;
  score: number;
  label_status: LabelStatus;
  review_flag?: boolean;
}

export interface RiskScoresFilterParams {
  search?: string;
  state?: string;
  district?: string;
  status?: string;
  min_amount?: number | string;
  max_amount?: number | string;
  min_lag?: number | string;
  max_lag?: number | string;
  label_status?: string;
  flag_pct?: number;
}

export interface FilterOptionsResponse {
  states: string[];
  districts_by_state: Record<string, string[]>;
  districts: string[];
  statuses: string[];
  amount_min: number;
  amount_max: number;
  lag_min: number;
  lag_max: number;
}

export interface MethodComparisonItem {
  id: ScoringMethod;
  name: string;
  score: number;
  global_rank: number;
  percentile: number;
  direction: "higher_is_riskier" | "lower_is_riskier";
}

export interface WorkCompareResponse {
  work_index: number;
  total_works: number;
  methods: Record<ScoringMethod, MethodComparisonItem>;
}

export interface ReviewBudgetRecallRow {
  Detector: string;
  Detector_Column: string;
  Typology: string;
  Total_Injected: number;
  "Hits_Top 1%": number;
  "Recall_Top 1%": number;
  "Recall_Str_Top 1%": string;
  "Hits_Top 2%": number;
  "Recall_Top 2%": number;
  "Recall_Str_Top 2%": string;
  "Hits_Top 5%": number;
  "Recall_Top 5%": number;
  "Recall_Str_Top 5%": string;
  "Hits_Top 10%": number;
  "Recall_Top 10%": number;
  "Recall_Str_Top 10%": string;
}

export interface FusionSummaryRow {
  model: string;
  mean_auprc_p1?: number;
  mean_auprc_fgh?: number;
  diff_vs_bot_mean: number;
  diff_vs_bot_ci_95: string;
  pos_families_vs_bot?: string;
  pos_typ_fgh_vs_bot?: string;
  pos_typ_all_vs_bot?: string;
  beats_bot_pooled?: string;
  beats_bot_fgh?: string;
  diff_vs_hin_mean: number;
  diff_vs_hin_ci_95: string;
  pos_families_vs_hin?: string;
  pos_typ_fgh_vs_hin?: string;
  pos_typ_all_vs_hin?: string;
  beats_hin_pooled?: string;
  beats_hin_fgh?: string;
}

export interface EvaluationResponse {
  review_budget_recall: ReviewBudgetRecallRow[];
  p1_pooled_summary: FusionSummaryRow[];
  p2_bootstrap_primary_fgh: FusionSummaryRow[];
  limitations: string[];
}

export interface OverviewKpis {
  total_works: number;
  total_amount: number;
  avg_lag_days: number;
  flagged_count: number;
  flagged_amount: number;
  flagged_amount_share: number;
  flag_pct: number;
  method: string;
}

export interface StateOverviewItem {
  state: string;
  total_count: number;
  flagged_count: number;
  flagged_share: number;
  total_amount: number;
  flagged_amount: number;
}

export interface StatusOverviewItem {
  status: string;
  total_count: number;
  flagged_count: number;
  flagged_share: number;
  total_amount: number;
  flagged_amount: number;
}

export interface MonthlyTimelineItem {
  month: string;
  total_count: number;
  flagged_count: number;
  flagged_share: number;
  total_amount: number;
  flagged_amount: number;
}

export interface ScatterPoint {
  work_index: number;
  amount: number;
  lag: number;
  flagged: boolean;
  rank: number;
  state: string;
  status: string;
}

export interface OverviewResponse {
  kpis: OverviewKpis;
  by_state: StateOverviewItem[];
  by_status: StatusOverviewItem[];
  monthly_timeline: MonthlyTimelineItem[];
  scatter_points: ScatterPoint[];
}

export interface DistrictGeoItem {
  district: string;
  works: number;
  total_amount: number;
  flagged_count: number;
  flagged_share: number;
  primary_ida: string;
  is_small_group: boolean;
}

export interface StateGeoItem {
  state: string;
  works: number;
  total_amount: number;
  flagged_count: number;
  flagged_share: number;
  is_small_group: boolean;
  districts: DistrictGeoItem[];
}

export interface GeographyResponse {
  method: string;
  flag_pct: number;
  total_works: number;
  total_flagged: number;
  states: StateGeoItem[];
}

export interface AgencyWorkItem {
  work_index: number;
  global_rank: number;
  review_flag: boolean;
  MP: string;
  IDA: string;
  amount: number;
  sanction_lag_days: number;
  status: string;
  category: string;
  agency_median_amount: number;
  agency_median_lag: number;
}

export interface AgencyResponse {
  ida: string;
  district: string;
  state: string;
  works: number;
  total_amount: number;
  median_amount: number;
  median_lag: number;
  flagged_count: number;
  flagged_share: number;
  is_small_group: boolean;
  status_mix: { status: string; count: number; share: number }[];
  works_data: AgencyWorkItem[];
  method: string;
  flag_pct: number;
}

export interface RiskScoresResponse {
  data: RiskRow[];
  page: number;
  page_size: number;
  total: number;
  method?: string;
  method_note?: string;
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

export async function fetchLabelTotals(): Promise<Sourced<LabelTotals>> {
  try {
    const data = await getJson<LabelTotals>(`/label-totals`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: { s: 0, n: 0, skip: 0 },
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export interface HealthResponse {
  status: string;
  message: string;
  labelling_enabled: boolean;
}

export async function fetchHealth(): Promise<Sourced<HealthResponse>> {
  try {
    const data = await getJson<HealthResponse>(`/health`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: { status: "offline", message: "backend unreachable", labelling_enabled: false },
      source: "mock",
      error: (e as Error).message,
    };
  }
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

export const DEFAULT_METHODS: MethodInfo[] = [
  {
    id: "gat",
    name: "Graph Attention Network (GAT)",
    direction: "higher_is_riskier",
    note: "Reconstruction error: flags statistically unusual works. Favours rare values; not evidence of misconduct.",
  },
  {
    id: "siamese",
    name: "Siamese Network",
    direction: "lower_is_riskier",
    note: "Similarity to 51 analyst-labelled works (drawn from GAT-ranked lists, so biased).",
  },
  {
    id: "community",
    name: "Community Anomaly Scorer",
    direction: "higher_is_riskier",
    note: "Several MPs sanctioning similar amounts at one agency within 30 days. Round-sum coincidences are common.",
  },
  {
    id: "fusion",
    name: "Learned Fusion",
    direction: "higher_is_riskier",
    note: "Experimental. Trained on synthetic fraud patterns, not validated on real fraud labels. A rank-ordering score, not a probability. It disagreed with analyst labels (see report).",
  },
];

export async function fetchMethods(): Promise<Sourced<MethodsResponse>> {
  try {
    const data = await getJson<MethodsResponse>(`/methods`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: { methods: DEFAULT_METHODS },
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchRiskScores(
  method: ScoringMethod,
  page: number,
  pageSize: number,
  filters?: RiskScoresFilterParams,
): Promise<Sourced<RiskScoresResponse>> {
  try {
    const params = new URLSearchParams({
      method,
      page: String(page),
      page_size: String(pageSize),
    });
    if (filters?.search?.trim()) params.set("search", filters.search.trim());
    if (filters?.state?.trim()) params.set("state", filters.state.trim());
    if (filters?.district?.trim()) params.set("district", filters.district.trim());
    if (filters?.status?.trim()) params.set("status", filters.status.trim());
    if (filters?.min_amount !== undefined && filters.min_amount !== "" && !isNaN(Number(filters.min_amount))) {
      params.set("min_amount", String(filters.min_amount));
    }
    if (filters?.max_amount !== undefined && filters.max_amount !== "" && !isNaN(Number(filters.max_amount))) {
      params.set("max_amount", String(filters.max_amount));
    }
    if (filters?.min_lag !== undefined && filters.min_lag !== "" && !isNaN(Number(filters.min_lag))) {
      params.set("min_lag", String(filters.min_lag));
    }
    if (filters?.max_lag !== undefined && filters.max_lag !== "" && !isNaN(Number(filters.max_lag))) {
      params.set("max_lag", String(filters.max_lag));
    }
    if (filters?.label_status?.trim()) params.set("label_status", filters.label_status.trim());
    if (filters?.flag_pct !== undefined && filters.flag_pct !== null) {
      params.set("flag_pct", String(filters.flag_pct));
    }

    const data = await getJson<RiskScoresResponse>(`/risk-scores?${params.toString()}`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: mockRiskScores(method, page, pageSize, filters),
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchWorkCompare(workIndex: number): Promise<Sourced<WorkCompareResponse>> {
  try {
    const data = await getJson<WorkCompareResponse>(`/work/${workIndex}/compare`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: mockWorkCompare(workIndex),
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchEvaluation(): Promise<Sourced<EvaluationResponse>> {
  try {
    const data = await getJson<EvaluationResponse>(`/evaluation`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: mockEvaluation(),
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchOverview(
  method: ScoringMethod = "gat",
  flagPct: 1 | 5 | 10 = 5,
): Promise<Sourced<OverviewResponse>> {
  try {
    const data = await getJson<OverviewResponse>(`/overview?method=${method}&flag_pct=${flagPct}`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: mockOverview(method, flagPct),
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchGeography(
  method: ScoringMethod = "gat",
  flagPct: 1 | 5 | 10 = 5,
): Promise<Sourced<GeographyResponse>> {
  try {
    const data = await getJson<GeographyResponse>(`/geography?method=${method}&flag_pct=${flagPct}`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: mockGeography(method, flagPct),
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchAgency(
  ida: string,
  method: ScoringMethod = "gat",
  flagPct: 1 | 5 | 10 = 5,
): Promise<Sourced<AgencyResponse>> {
  try {
    const encoded = encodeURIComponent(ida);
    const data = await getJson<AgencyResponse>(`/agency/${encoded}?method=${method}&flag_pct=${flagPct}`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: mockAgency(ida, method, flagPct),
      source: "mock",
      error: (e as Error).message,
    };
  }
}

export async function fetchFilterOptions(): Promise<Sourced<FilterOptionsResponse>> {
  try {
    const data = await getJson<FilterOptionsResponse>(`/filter-options`);
    return { data, source: "live" };
  } catch (e) {
    return {
      data: {
        states: [],
        districts_by_state: {},
        districts: [],
        statuses: [],
        amount_min: 0,
        amount_max: 50000000,
        lag_min: 0,
        lag_max: 1000,
      },
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
    if (res.status === 409) {
      const data = await res.json().catch(() => ({}));
      const error = new Error(data.detail || "Already labeled");
      (error as any).status = 409;
      throw error;
    }
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const error = new Error(data.detail || `${res.status} ${res.statusText}`);
      (error as any).status = res.status;
      throw error;
    }
    return { data: (await res.json()) as LabelResponse, source: "live" };
  } catch (e: any) {
    if (e?.status === 409 || e?.message?.includes("already been labeled") || e?.message?.includes("Already labeled")) {
      throw e;
    }
    if (e?.status && e.status !== 0) {
      throw e;
    }
    throw new Error("Backend unreachable; label not saved");
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

const STATES = [
  "Karnataka", "Bihar", "West Bengal", "Gujarat", "Punjab", "Madhya Pradesh",
  "Kerala", "Tamil Nadu", "Maharashtra", "Uttar Pradesh", "Rajasthan", "Andhra Pradesh"
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

export const METHOD_NOTES: Record<ScoringMethod, string> = Object.fromEntries(
  DEFAULT_METHODS.map((m) => [m.id, m.note]),
) as Record<ScoringMethod, string>;

function mockRiskScores(
  method: ScoringMethod,
  page: number,
  pageSize: number,
  filters?: RiskScoresFilterParams,
): RiskScoresResponse {
  const offset = (page - 1) * pageSize;
  const bias = method === "gat" ? 0 : method === "siamese" ? 7 : method === "community" ? 14 : 21;
  const flagThreshold = Math.round(15000 * ((filters?.flag_pct ?? 5) / 100));
  const data: RiskRow[] = Array.from({ length: pageSize }, (_, i) => {
    const rank = offset + i + 1;
    const r = seeded(rank + bias);
    // For Siamese, lower_is_riskier, so lowest distance is at rank 1.
    // For others, higher_is_riskier.
    const score =
      method === "siamese"
        ? Math.min(10.0, 0.05 + rank * 0.0005 + r * 0.02)
        : Math.max(0.01, 0.995 - rank * 0.00006 - r * 0.02);
    return {
      work_index: 100000 + rank * 7 + bias,
      global_rank: rank,
      MP: at(MPS, rank + bias),
      IDA: at(IDAS, rank * 3 + bias),
      amount: Math.round((5 + r * 240) * 100000),
      sanction_lag_days: Math.round(4 + seeded(rank * 2 + bias) * 730),
      status: at(STATUSES, rank + bias),
      category: at(CATEGORIES, rank * 2 + bias),
      score,
      label_status: at(LABELS, rank + bias),
      review_flag: rank <= flagThreshold,
    };
  });
  return {
    data,
    page,
    page_size: pageSize,
    total: MOCK_TOTAL,
    method,
    method_note: METHOD_NOTES[method] ?? "",
  };
}

function mockWorkCompare(workIndex: number): WorkCompareResponse {
  const r1 = (workIndex % 1500) + 1;
  const r2 = ((workIndex * 3) % 1500) + 1;
  const r3 = ((workIndex * 5) % 2500) + 1;
  const r4 = ((workIndex * 7) % 2000) + 1;
  return {
    work_index: workIndex,
    total_works: 15000,
    methods: {
      gat: {
        id: "gat",
        name: "Graph Attention Network (GAT)",
        score: 1.45,
        global_rank: r1,
        percentile: Number(((r1 / 15000) * 100).toFixed(2)),
        direction: "higher_is_riskier",
      },
      siamese: {
        id: "siamese",
        name: "Siamese Network",
        score: 0.82,
        global_rank: r2,
        percentile: Number(((r2 / 15000) * 100).toFixed(2)),
        direction: "lower_is_riskier",
      },
      community: {
        id: "community",
        name: "Community Anomaly Scorer",
        score: 0.65,
        global_rank: r3,
        percentile: Number(((r3 / 15000) * 100).toFixed(2)),
        direction: "higher_is_riskier",
      },
      fusion: {
        id: "fusion",
        name: "Learned Fusion",
        score: 0.73,
        global_rank: r4,
        percentile: Number(((r4 / 15000) * 100).toFixed(2)),
        direction: "higher_is_riskier",
      },
    },
  };
}

function mockEvaluation(): EvaluationResponse {
  return {
    review_budget_recall: [
      {
        Detector: "GAT (Frozen)",
        Detector_Column: "gat_frozen",
        Typology: "ALL (Total)",
        Total_Injected: 430,
        "Hits_Top 1%": 25,
        "Recall_Top 1%": 0.058,
        "Recall_Str_Top 1%": "5.8% (25/430)",
        "Hits_Top 2%": 48,
        "Recall_Top 2%": 0.112,
        "Recall_Str_Top 2%": "11.2% (48/430)",
        "Hits_Top 5%": 81,
        "Recall_Top 5%": 0.188,
        "Recall_Str_Top 5%": "18.8% (81/430)",
        "Hits_Top 10%": 126,
        "Recall_Top 10%": 0.293,
        "Recall_Str_Top 10%": "29.3% (126/430)",
      },
      {
        Detector: "Community Scorer",
        Detector_Column: "community_score",
        Typology: "ALL (Total)",
        Total_Injected: 430,
        "Hits_Top 1%": 20,
        "Recall_Top 1%": 0.047,
        "Recall_Str_Top 1%": "4.7% (20/430)",
        "Hits_Top 2%": 40,
        "Recall_Top 2%": 0.093,
        "Recall_Str_Top 2%": "9.3% (40/430)",
        "Hits_Top 5%": 75,
        "Recall_Top 5%": 0.174,
        "Recall_Str_Top 5%": "17.4% (75/430)",
        "Hits_Top 10%": 110,
        "Recall_Top 10%": 0.256,
        "Recall_Str_Top 10%": "25.6% (110/430)",
      },
    ],
    p1_pooled_summary: [
      {
        model: "hgbc_full",
        mean_auprc_p1: 0.0672,
        diff_vs_bot_mean: 0.0401,
        diff_vs_bot_ci_95: "[+0.0259, +0.0581]",
        pos_families_vs_bot: "6/6",
        beats_bot_pooled: "YES",
        diff_vs_hin_mean: -0.0546,
        diff_vs_hin_ci_95: "[-0.0946, -0.0163]",
        pos_families_vs_hin: "3/6",
        beats_hin_pooled: "NO",
      },
      {
        model: "logistic_full",
        mean_auprc_p1: 0.0567,
        diff_vs_bot_mean: 0.0282,
        diff_vs_bot_ci_95: "[+0.0156, +0.0448]",
        pos_families_vs_bot: "6/6",
        beats_bot_pooled: "YES",
        diff_vs_hin_mean: -0.0664,
        diff_vs_hin_ci_95: "[-0.1030, -0.0293]",
        pos_families_vs_hin: "3/6",
        beats_hin_pooled: "NO",
      },
    ],
    p2_bootstrap_primary_fgh: [
      {
        model: "hgbc_full",
        mean_auprc_fgh: 0.0560,
        diff_vs_bot_mean: 0.0251,
        diff_vs_bot_ci_95: "[+0.0078, +0.0468]",
        pos_typ_fgh_vs_bot: "3/3",
        beats_bot_fgh: "YES",
        diff_vs_hin_mean: 0.0250,
        diff_vs_hin_ci_95: "[+0.0067, +0.0494]",
        pos_typ_fgh_vs_hin: "3/3",
        beats_hin_fgh: "YES",
      },
    ],
    limitations: [
      "No ground truth, and a 15,000-work sample: True misconduct ground truth does not exist in this 15,000-work sample; labels reflect offline triage inspection, not adjudicated findings.",
      "Synthetic generator signatures: Benchmarks evaluate synthetically injected episodes; models may detect artifactual signatures of the generator rather than real-world corruption.",
      "Typology F probably driven by host selection: Detection performance on Typology F is probably driven by host selection patterns rather than genuine anomaly detection.",
      "Typologies G and H caught by no detector: Subtle distributed patterns (Typologies G and H) are caught by virtually no detector across low triage budgets, remaining near random chance.",
      "Fusion disagreement with analyst labels: Learned fusion models were trained on synthetic fraud patterns and disagreed with offline analyst review labels (which were drawn from GAT-ranked lists).",
    ],
  };
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

function mockOverview(method: ScoringMethod, flagPct: 1 | 5 | 10): OverviewResponse {
  const total_works = 15000;
  const flagged_count = Math.round(total_works * (flagPct / 100));
  // Neutral placeholder figures for demo mode (real totals only read from live API)
  const total_amount = 0;
  const flagged_amount = 0;
  const avg_lag_days = 0;

  const by_state: StateOverviewItem[] = STATES.map((state, i) => {
    const total_count = Math.round(400 + seeded(i + 1) * 600);
    const flg = Math.round(total_count * (flagPct / 100) * (0.8 + seeded(i + 2) * 0.5));
    return {
      state,
      total_count,
      flagged_count: flg,
      flagged_share: Number(((flg / total_count) * 100).toFixed(2)),
      total_amount: total_count * 500000,
      flagged_amount: flg * 650000,
    };
  }).sort((a, b) => b.flagged_share - a.flagged_share);

  const by_status: StatusOverviewItem[] = STATUSES.map((status, i) => {
    const total_count = Math.round(1500 + seeded(i + 10) * 3000);
    const flg = Math.round(total_count * (flagPct / 100) * (0.8 + seeded(i + 20) * 0.4));
    return {
      status,
      total_count,
      flagged_count: flg,
      flagged_share: Number(((flg / total_count) * 100).toFixed(2)),
      total_amount: total_count * 500000,
      flagged_amount: flg * 600000,
    };
  }).sort((a, b) => b.flagged_share - a.flagged_share);

  const months = [
    "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
  ];
  const monthly_timeline: MonthlyTimelineItem[] = months.map((month, i) => {
    const total_count = Math.round(400 + seeded(i + 30) * 1500);
    const flg = Math.round(total_count * (flagPct / 100) * (0.7 + seeded(i + 40) * 0.6));
    return {
      month,
      total_count,
      flagged_count: flg,
      flagged_share: Number(((flg / total_count) * 100).toFixed(2)),
      total_amount: total_count * 500000,
      flagged_amount: flg * 650000,
    };
  });

  const scatter_points: ScatterPoint[] = Array.from({ length: 1000 }, (_, i) => {
    const isFlg = i < flagged_count / 2;
    return {
      work_index: i,
      amount: Math.round(50000 + seeded(i * 3) * 4500000),
      lag: Math.round(10 + seeded(i * 7) * 400),
      flagged: isFlg,
      rank: isFlg ? i + 1 : 1000 + i,
      state: at(STATES, i % STATES.length),
      status: at(STATUSES, i % STATUSES.length),
    };
  });

  return {
    kpis: {
      total_works,
      total_amount,
      avg_lag_days,
      flagged_count,
      flagged_amount,
      flagged_amount_share: 0,
      flag_pct: flagPct,
      method,
    },
    by_state,
    by_status,
    monthly_timeline,
    scatter_points,
  };
}

function mockGeography(method: ScoringMethod, flagPct: 1 | 5 | 10): GeographyResponse {
  const total_works = 15000;
  const total_flagged = Math.round(total_works * (flagPct / 100));

  const states: StateGeoItem[] = STATES.map((state, sIdx) => {
    const numDistricts = 4;
    const districts: DistrictGeoItem[] = Array.from({ length: numDistricts }, (_, dIdx) => {
      // Deterministically create some small groups (<20 works) and normal groups
      const isSmall = (sIdx + dIdx) % 3 === 0;
      const dWorks = isSmall ? Math.round(5 + seeded(sIdx * 10 + dIdx) * 12) : Math.round(80 + seeded(sIdx * 10 + dIdx) * 300);
      const dFlagged = Math.round(dWorks * (flagPct / 100) * (0.8 + seeded(sIdx * 7 + dIdx) * 0.4));
      const distName = `${state}_Dist_${dIdx + 1}`;
      return {
        district: distName,
        works: dWorks,
        total_amount: dWorks * 500000,
        flagged_count: dFlagged,
        flagged_share: Number(((dFlagged / dWorks) * 100).toFixed(2)),
        primary_ida: `${distName}(COLLECTORATE_${distName}_IDA)`,
        is_small_group: dWorks < 20,
      };
    }).sort((a, b) => b.works - a.works);

    const sWorks = districts.reduce((acc, d) => acc + d.works, 0);
    const sAmount = districts.reduce((acc, d) => acc + d.total_amount, 0);
    const sFlagged = districts.reduce((acc, d) => acc + d.flagged_count, 0);
    const sShare = Number(((sFlagged / sWorks) * 100).toFixed(2));

    return {
      state,
      works: sWorks,
      total_amount: sAmount,
      flagged_count: sFlagged,
      flagged_share: sShare,
      is_small_group: sWorks < 20,
      districts,
    };
  }).sort((a, b) => b.works - a.works);

  return {
    method,
    flag_pct: flagPct,
    total_works,
    total_flagged,
    states,
  };
}

function mockAgency(ida: string, method: ScoringMethod, flagPct: 1 | 5 | 10): AgencyResponse {
  const works = 35;
  const flagged_count = Math.max(1, Math.round(works * (flagPct / 100)));
  const median_amount = 450000;
  const median_lag = 48;
  const total_amount = works * median_amount;

  const dist = ida.split("(")[0]?.trim() || "DHARWAD";
  const state = at(STATES, 0);

  const works_data: AgencyWorkItem[] = Array.from({ length: works }, (_, i) => {
    const isFlg = i < flagged_count;
    return {
      work_index: i + 100,
      global_rank: isFlg ? (i + 1) * 30 : 2000 + i * 50,
      review_flag: isFlg,
      MP: at(MPS, i),
      IDA: ida,
      amount: Math.round(median_amount * (0.6 + seeded(i * 3) * 0.9)),
      sanction_lag_days: Math.round(median_lag * (0.5 + seeded(i * 5) * 1.2)),
      status: at(STATUSES, i),
      category: at(CATEGORIES, i),
      agency_median_amount: median_amount,
      agency_median_lag: median_lag,
    };
  });

  return {
    ida,
    district: dist,
    state,
    works,
    total_amount,
    median_amount,
    median_lag,
    flagged_count,
    flagged_share: Number(((flagged_count / works) * 100).toFixed(2)),
    is_small_group: works < 20,
    status_mix: STATUSES.map((status, i) => ({
      status,
      count: Math.round(works / 4),
      share: 25.0,
    })),
    works_data,
    method,
    flag_pct: flagPct,
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
