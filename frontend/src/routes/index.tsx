import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useMutation, keepPreviousData } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import {
  fetchExplain,
  fetchGraph,
  fetchMethods,
  fetchRiskScores,
  fetchStats,
  fetchLabelTotals,
  postLabel,
  DEFAULT_METHODS,
  type LabelTotals,
  type LabelValue,
  type RiskRow,
  type ScoringMethod,
} from "@/lib/api";
import { API_BASE_URL } from "@/lib/config";
import { cn } from "@/lib/utils";
import { RiskTable } from "@/components/mplads/RiskTable";
import { StatsOverview } from "@/components/mplads/StatsOverview";
import { ExplainPanel } from "@/components/mplads/ExplainPanel";
import { NetworkGraph } from "@/components/mplads/NetworkGraph";

const TITLE = "MPLADS Fraud & Anomaly Detection";
const DESCRIPTION =
  "Review AI-flagged anomalies in MPLADS fund sanctioning data: risk-ranked works, neighbour attribution, network view and analyst labelling.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: `${TITLE} — Analyst Console` },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: `${TITLE} — Analyst Console` },
      { property: "og:description", content: DESCRIPTION },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Dashboard,
});

const PAGE_SIZE = 50;

function Dashboard() {
  const [method, setMethod] = useState<ScoringMethod>("gat");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<RiskRow | null>(null);
  const [totals, setTotals] = useState<LabelTotals>({ s: 0, n: 0, skip: 0 });
  const [rowFeedback, setRowFeedback] = useState<
    Record<number, { status: "success" | "duplicate" | "error"; message: string }>
  >({});

  const labelTotals = useQuery({
    queryKey: ["label-totals"],
    queryFn: fetchLabelTotals,
  });

  useEffect(() => {
    if (labelTotals.data?.data) {
      setTotals(labelTotals.data.data);
    }
  }, [labelTotals.data]);

  const methodsQuery = useQuery({
    queryKey: ["methods"],
    queryFn: fetchMethods,
  });

  const methodList = methodsQuery.data?.data.methods ?? DEFAULT_METHODS;
  const currentMethod =
    methodList.find((m) => m.id === method) ?? DEFAULT_METHODS.find((m) => m.id === method);

  const scores = useQuery({
    queryKey: ["risk-scores", method, page],
    queryFn: () => fetchRiskScores(method, page, PAGE_SIZE),
    placeholderData: keepPreviousData,
  });

  const methodNote = scores.data?.data.method_note || currentMethod?.note;

  const graph = useQuery({ queryKey: ["graph", 50], queryFn: () => fetchGraph(50) });

  const stats = useQuery({ queryKey: ["stats"], queryFn: () => fetchStats() });

  const explain = useQuery({
    queryKey: ["explain", selected?.work_index],
    queryFn: () => fetchExplain(selected!.work_index),
    enabled: selected !== null && method === "gat",
  });

  const isAnyMock =
    scores.data?.source === "mock" ||
    methodsQuery.data?.source === "mock" ||
    graph.data?.source === "mock" ||
    stats.data?.source === "mock" ||
    explain.data?.source === "mock" ||
    labelTotals.data?.source === "mock";

  const label = useMutation({
    mutationFn: ({ workIndex, value }: { workIndex: number; value: LabelValue }) => {
      if (isAnyMock) {
        return Promise.reject(new Error("Labelling disabled in demo mode."));
      }
      return postLabel(workIndex, value);
    },
    onSuccess: (res, variables) => {
      if (res.data?.totals) {
        setTotals(res.data.totals);
      }
      labelTotals.refetch();
      setRowFeedback((prev) => ({
        ...prev,
        [variables.workIndex]: {
          status: "success",
          message:
            variables.value === "s"
              ? "Marked Suspicious"
              : variables.value === "n"
              ? "Marked Normal"
              : "Skipped",
        },
      }));
      scores.refetch();
      stats.refetch();
    },
    onError: (error: any, variables) => {
      if (
        error?.status === 409 ||
        error?.message?.toLowerCase().includes("already") ||
        error?.message?.includes("409")
      ) {
        setRowFeedback((prev) => ({
          ...prev,
          [variables.workIndex]: {
            status: "duplicate",
            message: "Already labeled",
          },
        }));
      } else {
        setRowFeedback((prev) => ({
          ...prev,
          [variables.workIndex]: {
            status: "error",
            message: error?.message || "Failed to label",
          },
        }));
      }
    },
  });

  const rows = scores.data?.data.data ?? [];
  const total = scores.data?.data.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const flaggedCount = stats.data?.data
    ? stats.data.data.label_counts.training_label + stats.data.data.label_counts.gat_top
    : 62;

  const onLabel = (workIndex: number, value: LabelValue) => {
    if (isAnyMock) return;
    label.mutate({ workIndex, value });
  };

  return (
    <div className="min-h-screen bg-background flex flex-col justify-between">
      <div>
        {isAnyMock && (
          <div className="border-b border-alert bg-alert/15 px-6 py-2.5 text-center text-xs font-semibold text-alert">
            Demo data: backend unreachable. Nothing on this page is real output.
          </div>
        )}

        <header className="border-b border-border bg-card">
          <div className="mx-auto flex max-w-[1600px] flex-wrap items-end justify-between gap-6 px-6 py-5">
            <div>
              <p className="text-[11px] tracking-[0.18em] text-muted-foreground uppercase">
                Member of Parliament Local Area Development Scheme
              </p>
              <h1 className="mt-1 text-2xl font-semibold">{TITLE}</h1>
              <p className="mt-1 text-xs text-muted-foreground">
                Analyst review console · endpoint <span className="font-mono">{API_BASE_URL}</span>
              </p>
            </div>
            <dl className="flex gap-6 border border-border bg-secondary/60 px-5 py-3 text-center">
              {(
                [
                  ["Suspicious", totals.s, "text-alert"],
                  ["Normal", totals.n, "text-ok"],
                  ["Skipped", totals.skip, "text-muted-foreground"],
                ] as const
              ).map(([k, v, cls]) => (
                <div key={k}>
                  <dt className="text-[11px] tracking-wider text-muted-foreground uppercase">{k}</dt>
                  <dd className={cn("mt-0.5 font-mono text-xl tabular-nums", cls)}>{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </header>

        <main className="mx-auto grid max-w-[1600px] gap-5 px-6 py-6 xl:grid-cols-[minmax(0,1fr)_420px]">
          <div className="xl:col-span-2">
            <StatsOverview
              stats={stats.data?.data ?? null}
              isLoading={stats.isLoading}
              isMock={stats.data?.source === "mock"}
              error={stats.data?.error}
              flaggedCount={flaggedCount}
            />
          </div>

          <section className="border border-border bg-card">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border px-5 py-3">
              <div>
                <h2 className="text-base font-semibold">Risk-ranked sanctioned works</h2>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {total.toLocaleString("en-IN")} records · page {page} of{" "}
                  {pageCount.toLocaleString("en-IN")}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] tracking-wider text-muted-foreground uppercase">
                  Scoring Method
                </span>
                <div className="flex border border-border">
                  {methodList.map((m) => {
                    const isSelected = method === m.id;
                    return (
                      <button
                        key={m.id}
                        onClick={() => {
                          setMethod(m.id);
                          setPage(1);
                          setSelected(null);
                        }}
                        className={cn(
                          "px-3 py-1.5 text-xs font-medium cursor-pointer transition-colors flex items-center gap-1.5",
                          isSelected
                            ? "bg-navy text-navy-foreground"
                            : "text-muted-foreground hover:bg-muted",
                        )}
                      >
                        <span>
                          {m.id === "gat"
                            ? "GAT"
                            : m.id === "siamese"
                            ? "Siamese"
                            : m.id === "community"
                            ? "Community"
                            : "Fusion"}
                        </span>
                        {m.id === "fusion" && (
                          <span
                            className={cn(
                              "rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
                              isSelected
                                ? "bg-amber-400 text-amber-950"
                                : "bg-amber-500/20 text-amber-600 dark:text-amber-400",
                            )}
                          >
                            Experimental
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Method Note Banner */}
            {methodNote && (
              <div
                className={cn(
                  "flex items-start gap-2.5 border-b px-5 py-3 text-xs",
                  method === "fusion"
                    ? "border-amber-500/40 bg-amber-500/10 text-amber-900 dark:text-amber-200"
                    : "border-border bg-secondary/50 text-foreground/80",
                )}
              >
                <span
                  className={cn(
                    "font-bold text-sm leading-none mt-0.5",
                    method === "fusion"
                      ? "text-amber-600 dark:text-amber-400"
                      : "text-muted-foreground",
                  )}
                  aria-hidden="true"
                >
                  {method === "fusion" ? "⚠️" : "ℹ️"}
                </span>
                <div className="flex-1">
                  <span
                    className={cn(
                      "font-semibold",
                      method === "fusion"
                        ? "text-amber-800 dark:text-amber-300"
                        : "text-foreground",
                    )}
                  >
                    {method === "fusion" ? "Method Note (Experimental): " : "Method Note: "}
                  </span>
                  <span>{methodNote}</span>
                  {currentMethod?.direction === "lower_is_riskier" && (
                    <span className="ml-1.5 font-medium text-muted-foreground">
                      (Note: lower scores indicate higher risk)
                    </span>
                  )}
                </div>
              </div>
            )}

            {scores.isLoading ? (
              <p className="px-5 py-12 text-sm text-muted-foreground">Loading triage scores…</p>
            ) : (
              <RiskTable
                rows={rows}
                offset={(page - 1) * PAGE_SIZE}
                total={total}
                selected={selected?.work_index ?? null}
                isLoading={scores.isFetching}
                isDemo={isAnyMock}
                direction={currentMethod?.direction}
                rowFeedback={rowFeedback}
                onSelect={setSelected}
                onLabel={onLabel}
              />
            )}

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-5 py-3 text-xs">
              <span className="text-muted-foreground">
                Rows {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, total)}
              </span>
              <div className="flex items-center gap-2">
                <PagerButton disabled={page === 1} onClick={() => setPage(1)} label="First" />
                <PagerButton
                  disabled={page === 1}
                  onClick={() => setPage((p) => p - 1)}
                  label="Previous"
                />
                <input
                  type="number"
                  min={1}
                  max={pageCount}
                  value={page}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (v >= 1 && v <= pageCount) setPage(v);
                  }}
                  className="w-20 border border-border bg-background px-2 py-1 text-center font-mono tabular-nums"
                />
                <PagerButton
                  disabled={page >= pageCount}
                  onClick={() => setPage((p) => p + 1)}
                  label="Next"
                />
                <PagerButton
                  disabled={page >= pageCount}
                  onClick={() => setPage(pageCount)}
                  label="Last"
                />
              </div>
            </div>
          </section>

          <aside className="flex flex-col gap-5">
            <ExplainPanel
              row={selected}
              method={method}
              explain={explain.data?.data ?? null}
              isLoading={explain.isFetching}
              isMock={explain.data?.source === "mock"}
              isDemo={isAnyMock}
              feedback={selected ? rowFeedback[selected.work_index] : null}
              onLabel={onLabel}
            />
          </aside>

          <div className="xl:col-span-2">
            <NetworkGraph
              graph={graph.data?.data ?? null}
              isLoading={graph.isLoading}
              isMock={graph.data?.source === "mock"}
              flaggedCount={flaggedCount}
              selectedWorkIndex={selected?.work_index ?? null}
              onSelectEdge={(workIndex) => {
                const row = rows.find((r) => r.work_index === workIndex);
                if (row) setSelected(row);
              }}
            />
          </div>
        </main>
      </div>

      <footer className="border-t border-border bg-card py-5 text-center text-xs text-muted-foreground">
        Flags are leads for review, not confirmed fraud.
      </footer>
    </div>
  );
}

function PagerButton({
  disabled,
  onClick,
  label,
}: {
  disabled: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className="border border-border px-2.5 py-1 hover:bg-muted disabled:opacity-40 disabled:hover:bg-transparent"
    >
      {label}
    </button>
  );
}
