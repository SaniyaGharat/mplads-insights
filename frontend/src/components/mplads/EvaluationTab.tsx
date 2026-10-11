import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { fetchEvaluation, type EvaluationResponse, type ReviewBudgetRecallRow } from "@/lib/api";
import { cn } from "@/lib/utils";

const SAGE = "oklch(0.6 0.05 155)";
const NAVY = "oklch(0.36 0.055 255)";
const CLAY = "oklch(0.53 0.08 55)";
const AMBER = "oklch(0.65 0.12 75)";
const PURPLE = "oklch(0.5 0.1 300)";
const GRID = "oklch(0.885 0.008 90)";
const AXIS = "oklch(0.5 0.02 255)";

const DETECTOR_COLORS: Record<string, string> = {
  "GAT (Frozen)": NAVY,
  "GAT (Augmented)": SAGE,
  "Community Scorer": AMBER,
  "Isolation Forest": PURPLE,
  "Robust Z-score": CLAY,
};

const MODEL_NAMES: Record<string, string> = {
  logistic_full: "Logistic Regression (Full features)",
  logistic_detectors: "Logistic Regression (Detectors only)",
  logistic_tabular: "Logistic Regression (Tabular baseline)",
  hgbc_full: "Histogram GBDT (Full features)",
  hgbc_detectors: "Histogram GBDT (Detectors only)",
  hgbc_tabular: "Histogram GBDT (Tabular baseline)",
};

export function EvaluationTab() {
  const evalQuery = useQuery({
    queryKey: ["evaluation"],
    queryFn: fetchEvaluation,
  });

  const [selectedTypology, setSelectedTypology] = useState<string>("ALL (Total)");
  const [fusionPhase, setFusionPhase] = useState<"p1" | "p2">("p1");

  const data: EvaluationResponse | null = evalQuery.data?.data ?? null;

  // Typology options
  const typologies = useMemo(() => {
    if (!data?.review_budget_recall) return ["ALL (Total)"];
    const set = new Set(data.review_budget_recall.map((r) => r.Typology));
    const list = Array.from(set);
    list.sort((a, b) => {
      if (a === "ALL (Total)") return -1;
      if (b === "ALL (Total)") return 1;
      return a.localeCompare(b);
    });
    return list;
  }, [data?.review_budget_recall]);

  // Chart data for selected typology
  const chartData = useMemo(() => {
    if (!data?.review_budget_recall) return [];
    const rows = data.review_budget_recall.filter((r) => r.Typology === selectedTypology);

    const budgetLevels = [
      { key: "1%", label: "Top 1%", pctKey: "Recall_Top 1%" as const, strKey: "Recall_Str_Top 1%" as const, randomVal: 1.0 },
      { key: "2%", label: "Top 2%", pctKey: "Recall_Top 2%" as const, strKey: "Recall_Str_Top 2%" as const, randomVal: 2.0 },
      { key: "5%", label: "Top 5%", pctKey: "Recall_Top 5%" as const, strKey: "Recall_Str_Top 5%" as const, randomVal: 5.0 },
      { key: "10%", label: "Top 10%", pctKey: "Recall_Top 10%" as const, strKey: "Recall_Str_Top 10%" as const, randomVal: 10.0 },
    ];

    return budgetLevels.map((lvl) => {
      const point: Record<string, any> = {
        budget: lvl.label,
        "Random Baseline": lvl.randomVal,
        "Random Baseline_str": `${lvl.randomVal.toFixed(1)}% (random draw)`,
      };
      for (const row of rows) {
        const recallVal = (row[lvl.pctKey] || 0) * 100;
        point[row.Detector] = Number(recallVal.toFixed(2));
        point[`${row.Detector}_str`] = row[lvl.strKey];
      }
      return point;
    });
  }, [data?.review_budget_recall, selectedTypology]);

  // Detector list for lines
  const detectors = useMemo(() => {
    if (!data?.review_budget_recall) return [];
    const set = new Set(data.review_budget_recall.map((r) => r.Detector));
    return Array.from(set);
  }, [data?.review_budget_recall]);

  const currentTypologyRows = useMemo(() => {
    if (!data?.review_budget_recall) return [];
    return data.review_budget_recall.filter((r) => r.Typology === selectedTypology);
  }, [data?.review_budget_recall, selectedTypology]);

  if (evalQuery.isLoading) {
    return (
      <div className="border border-border bg-card p-12 text-center text-sm text-muted-foreground">
        Loading evaluation benchmarks and model comparison data…
      </div>
    );
  }

  if (evalQuery.isError || !data) {
    return (
      <div className="border border-destructive/40 bg-destructive/10 p-6 text-sm text-destructive">
        Failed to load evaluation data: {evalQuery.error?.message || "Unknown error"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="border border-border bg-card p-5">
        <h2 className="text-lg font-semibold">Offline Benchmark Evaluation (Synthetic Injected Episodes)</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          These are results on the synthetic benchmark, where the oracle is the injected episodes (Typologies A–H).
          This is not ground truth misconduct. Scores evaluate detection of synthetic injection patterns under controlled budgets.
        </p>
      </div>

      {/* Section 1: Recall vs Review Budget */}
      <div className="border border-border bg-card">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border px-5 py-3">
          <div>
            <h3 className="text-base font-semibold">Recall vs. Review Budget</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Proportion of injected anomalies identified within top 1%, 2%, 5%, and 10% triage review budgets, compared against a Random Baseline.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] tracking-wider text-muted-foreground uppercase">
              Anomaly Typology
            </span>
            <select
              value={selectedTypology}
              onChange={(e) => setSelectedTypology(e.target.value)}
              className="border border-border bg-background px-3 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            >
              {typologies.map((t) => (
                <option key={t} value={t}>
                  {t === "ALL (Total)" ? "ALL (Total - 430 injected)" : `Typology ${t}`}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="p-5">
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                <XAxis
                  dataKey="budget"
                  stroke={GRID}
                  tick={{ fill: AXIS, fontSize: 12 }}
                  tickLine={false}
                />
                <YAxis
                  stroke={GRID}
                  tick={{ fill: AXIS, fontSize: 12 }}
                  tickLine={false}
                  domain={[0, 100]}
                  unit="%"
                  label={{
                    value: "Recall (%)",
                    angle: -90,
                    position: "insideLeft",
                    fill: AXIS,
                    fontSize: 12,
                  }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: `1px solid ${GRID}`,
                    fontSize: "12px",
                  }}
                  formatter={(value: any, name: any, item: any) => [
                    `${value}% (${item.payload[`${name}_str`]?.split("(")[1] || ""}`,
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                {detectors.map((d) => (
                  <Line
                    key={d}
                    type="monotone"
                    dataKey={d}
                    stroke={DETECTOR_COLORS[d] || "#64748b"}
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                ))}
                <Line
                  type="monotone"
                  dataKey="Random Baseline"
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Breakdown Table for Selected Typology */}
          <div className="mt-6 overflow-x-auto border-t border-border pt-4">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
              Recall Breakdown — {selectedTypology} (
              {currentTypologyRows[0]?.Total_Injected ?? 0} total injected)
            </h4>
            <table className="w-full text-left text-xs border border-border">
              <thead>
                <tr className="border-b border-border bg-secondary/50 text-[11px] text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Detector</th>
                  <th className="px-3 py-2 font-medium text-right">Top 1% Budget</th>
                  <th className="px-3 py-2 font-medium text-right">Top 2% Budget</th>
                  <th className="px-3 py-2 font-medium text-right">Top 5% Budget</th>
                  <th className="px-3 py-2 font-medium text-right">Top 10% Budget</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {currentTypologyRows.map((r) => (
                  <tr key={r.Detector} className="hover:bg-muted/40">
                    <td className="px-3 py-2 font-medium flex items-center gap-2">
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: DETECTOR_COLORS[r.Detector] || "#64748b" }}
                      />
                      {r.Detector}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {r["Recall_Str_Top 1%"]}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {r["Recall_Str_Top 2%"]}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {r["Recall_Str_Top 5%"]}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {r["Recall_Str_Top 10%"]}
                    </td>
                  </tr>
                ))}
                {/* Random Baseline Row */}
                <tr className="bg-secondary/20 font-medium text-muted-foreground">
                  <td className="px-3 py-2 flex items-center gap-2">
                    <span className="inline-block w-2.5 h-0.5 border-t-2 border-dashed border-slate-400" />
                    Random Baseline (Expected)
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">1.00%</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">2.00%</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">5.00%</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">10.00%</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Honest Benchmark Observations Box */}
          <div className="mt-4 rounded border border-border/80 bg-secondary/30 p-3 text-xs text-muted-foreground space-y-1.5">
            <div className="font-semibold text-foreground flex items-center gap-1.5">
              <span>💡</span> Benchmark Findings & Honest Interpretations:
            </div>
            <p>
              • <strong>Random Baseline:</strong> Reviewing <em>B%</em> of works at random yields an expected <em>B%</em> recall (1% at 1%, 2% at 2%, 5% at 5%, 10% at 10%).
            </p>
            <p>
              • <strong>Typology G Near Chance:</strong> Typology G is near chance across detectors (detectors struggle to separate subtle distributed coordination).
            </p>
            <p>
              • <strong>1% Review Budget at ~0:</strong> At a tight 1% review budget, GAT and Isolation Forest achieve approximately 0% recall across multiple typologies.
            </p>
            <p>
              • <strong>Synthetic Oracle:</strong> Injected episodes serve as the oracle for these recall figures; results do not represent adjudicated real-world fraud.
            </p>
          </div>
        </div>
      </div>

      {/* Section 2: Fusion Benchmark Results */}
      <div className="border border-border bg-card">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border px-5 py-3">
          <div>
            <h3 className="text-base font-semibold">Multi-Model Fusion Benchmarks</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Statistical evaluation of supervised ensemble fusion vs. Bag-of-Tokens (BoT) and Heterogeneous Graph (HIN) baselines.
            </p>
          </div>
          <div className="flex border border-border">
            <button
              onClick={() => setFusionPhase("p1")}
              className={cn(
                "px-3 py-1.5 text-xs font-medium cursor-pointer transition-colors",
                fusionPhase === "p1"
                  ? "bg-navy text-navy-foreground"
                  : "text-muted-foreground hover:bg-muted"
              )}
            >
              Phase 1: Pooled (6 Families)
            </button>
            <button
              onClick={() => setFusionPhase("p2")}
              className={cn(
                "px-3 py-1.5 text-xs font-medium cursor-pointer transition-colors",
                fusionPhase === "p2"
                  ? "bg-navy text-navy-foreground"
                  : "text-muted-foreground hover:bg-muted"
              )}
            >
              Phase 2: Bootstrap Primary (F, G, H)
            </button>
          </div>
        </div>

        <div className="p-5 overflow-x-auto">
          {fusionPhase === "p1" ? (
            <div>
              <p className="text-xs text-muted-foreground mb-3">
                Pooled evaluation across 6 anomaly typologies. Metrics include mean AUPRC and 95% bootstrap confidence intervals for performance differences.
              </p>
              <table className="w-full text-left text-xs border border-border">
                <thead>
                  <tr className="border-b border-border bg-secondary/50 text-[11px] text-muted-foreground">
                    <th className="px-3 py-2 font-medium">Model</th>
                    <th className="px-3 py-2 font-medium text-right">Mean AUPRC</th>
                    <th className="px-3 py-2 font-medium text-right">Δ vs BoT (Mean)</th>
                    <th className="px-3 py-2 font-medium text-center">95% CI vs BoT</th>
                    <th className="px-3 py-2 font-medium text-center">Beats BoT</th>
                    <th className="px-3 py-2 font-medium text-right">Δ vs HIN (Mean)</th>
                    <th className="px-3 py-2 font-medium text-center">95% CI vs HIN</th>
                    <th className="px-3 py-2 font-medium text-center">Beats HIN</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {data.p1_pooled_summary.map((row) => (
                    <tr key={row.model} className="hover:bg-muted/40">
                      <td className="px-3 py-2 font-medium">
                        {MODEL_NAMES[row.model] || row.model}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {((row.mean_auprc_p1 ?? 0) * 100).toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {row.diff_vs_bot_mean >= 0 ? "+" : ""}
                        {(row.diff_vs_bot_mean * 100).toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
                        {row.diff_vs_bot_ci_95}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={cn(
                            "inline-block rounded px-2 py-0.5 text-[10px] font-semibold",
                            row.beats_bot_pooled === "YES"
                              ? "bg-ok/15 text-ok"
                              : "bg-muted text-muted-foreground"
                          )}
                        >
                          {row.beats_bot_pooled}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {row.diff_vs_hin_mean >= 0 ? "+" : ""}
                        {(row.diff_vs_hin_mean * 100).toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
                        {row.diff_vs_hin_ci_95}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={cn(
                            "inline-block rounded px-2 py-0.5 text-[10px] font-semibold",
                            row.beats_hin_pooled === "YES"
                              ? "bg-ok/15 text-ok"
                              : "bg-muted text-muted-foreground"
                          )}
                        >
                          {row.beats_hin_pooled}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div>
              <p className="text-xs text-muted-foreground mb-3">
                Bootstrap evaluation focused on primary typologies F, G, and H with 95% confidence intervals.
              </p>
              <table className="w-full text-left text-xs border border-border">
                <thead>
                  <tr className="border-b border-border bg-secondary/50 text-[11px] text-muted-foreground">
                    <th className="px-3 py-2 font-medium">Model</th>
                    <th className="px-3 py-2 font-medium text-right">Mean AUPRC (FGH)</th>
                    <th className="px-3 py-2 font-medium text-right">Δ vs BoT (Mean)</th>
                    <th className="px-3 py-2 font-medium text-center">95% CI vs BoT</th>
                    <th className="px-3 py-2 font-medium text-center">Beats BoT</th>
                    <th className="px-3 py-2 font-medium text-right">Δ vs HIN (Mean)</th>
                    <th className="px-3 py-2 font-medium text-center">95% CI vs HIN</th>
                    <th className="px-3 py-2 font-medium text-center">Beats HIN</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {data.p2_bootstrap_primary_fgh.map((row) => (
                    <tr key={row.model} className="hover:bg-muted/40">
                      <td className="px-3 py-2 font-medium">
                        {MODEL_NAMES[row.model] || row.model}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {((row.mean_auprc_fgh ?? 0) * 100).toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {row.diff_vs_bot_mean >= 0 ? "+" : ""}
                        {(row.diff_vs_bot_mean * 100).toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
                        {row.diff_vs_bot_ci_95}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={cn(
                            "inline-block rounded px-2 py-0.5 text-[10px] font-semibold",
                            row.beats_bot_fgh === "YES"
                              ? "bg-ok/15 text-ok"
                              : "bg-muted text-muted-foreground"
                          )}
                        >
                          {row.beats_bot_fgh}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">
                        {row.diff_vs_hin_mean >= 0 ? "+" : ""}
                        {(row.diff_vs_hin_mean * 100).toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
                        {row.diff_vs_hin_ci_95}
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={cn(
                            "inline-block rounded px-2 py-0.5 text-[10px] font-semibold",
                            row.beats_hin_fgh === "YES"
                              ? "bg-ok/15 text-ok"
                              : "bg-muted text-muted-foreground"
                          )}
                        >
                          {row.beats_hin_fgh}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Baseline Selection & Hindsight Caveat */}
          <div className="mt-4 rounded border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-900 dark:text-amber-200 space-y-1.5">
            <div className="font-semibold text-amber-800 dark:text-amber-300 flex items-center gap-1.5">
              <span>⚠️</span> Baseline Selection & Hindsight Caveat:
            </div>
            <p>
              The tables present performance vs. both the <strong>train-selected baseline</strong> (Bag-of-Tokens / BoT) and the <strong>heterogeneous graph baseline</strong> (HIN).
            </p>
            <p>
              Note that while models appear to "beat the baseline" on pooled train-selected benchmarks, this <strong>does not represent a hindsight baseline</strong>. Beating the train-selected baseline does not eliminate model selection bias or guarantee outperformance against hindsight optimal baseline choices.
            </p>
          </div>
        </div>
      </div>

      {/* Section 3: Limitations Box */}
      <div className="border border-border bg-card p-5">
        <div className="flex items-start gap-3">
          <span className="text-lg leading-none mt-0.5" aria-hidden="true">
            📋
          </span>
          <div className="space-y-3">
            <div>
              <h3 className="text-sm font-semibold">
                Methodological Limitations & Audit Usage Notice
              </h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Crucial caveats regarding score interpretation, benchmark boundaries, and domain applicability:
              </p>
            </div>
            <ul className="space-y-2 text-xs text-foreground/80 list-disc list-inside">
              {data.limitations.map((lim, idx) => (
                <li key={idx} className="leading-relaxed">
                  {lim}
                </li>
              ))}
            </ul>
            <div className="border-t border-border/60 pt-3 text-[11px] text-muted-foreground italic">
              Notice: Model scores and triage flags are intended solely for exploratory audit prioritisation and workflow triage. They do not constitute findings of administrative misconduct or ground truth determinations.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
