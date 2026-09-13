import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useMutation, keepPreviousData } from "@tanstack/react-query";
import { useState } from "react";
import {
  fetchExplain,
  fetchGraph,
  fetchRiskScores,
  fetchStats,
  postLabel,
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

  const scores = useQuery({
    queryKey: ["risk-scores", method, page],
    queryFn: () => fetchRiskScores(method, page, PAGE_SIZE),
    placeholderData: keepPreviousData,
  });

  const graph = useQuery({ queryKey: ["graph", 50], queryFn: () => fetchGraph(50) });

  const stats = useQuery({ queryKey: ["stats"], queryFn: () => fetchStats() });

  const explain = useQuery({
    queryKey: ["explain", selected?.work_index],
    queryFn: () => fetchExplain(selected!.work_index),
    enabled: selected !== null,
  });

  const label = useMutation({
    mutationFn: ({ workIndex, value }: { workIndex: number; value: LabelValue }) =>
      postLabel(workIndex, value),
    onSuccess: (res) => setTotals(res.data.totals),
  });

  const rows = scores.data?.data.data ?? [];
  const total = scores.data?.data.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const isMock = scores.data?.source === "mock";

  const onLabel = (workIndex: number, value: LabelValue) => label.mutate({ workIndex, value });

  return (
    <div className="min-h-screen bg-background">
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

      {isMock && (
        <div className="border-b border-alert/30 bg-alert/8">
          <p className="mx-auto max-w-[1600px] px-6 py-2 text-xs text-alert">
            Live API unreachable ({scores.data?.error ?? "network error"}). Showing sample records
            so the interface can be reviewed. Update the base URL in{" "}
            <span className="font-mono">src/lib/config.ts</span> once the backend is deployed.
          </p>
        </div>
      )}

      <main className="mx-auto grid max-w-[1600px] gap-5 px-6 py-6 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div className="xl:col-span-2">
          <StatsOverview
            stats={stats.data?.data ?? null}
            isLoading={stats.isLoading}
            isMock={stats.data?.source === "mock"}
            error={stats.data?.error}
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
                Scoring
              </span>
              <div className="flex border border-border">
                {(["gat", "siamese"] as ScoringMethod[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => {
                      setMethod(m);
                      setPage(1);
                    }}
                    className={cn(
                      "px-3 py-1.5 text-xs font-medium",
                      method === m
                        ? "bg-navy text-navy-foreground"
                        : "text-muted-foreground hover:bg-muted",
                    )}
                  >
                    {m === "gat" ? "GAT" : "Siamese"}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {scores.isLoading ? (
            <p className="px-5 py-12 text-sm text-muted-foreground">Loading risk scores…</p>
          ) : (
            <RiskTable
              rows={rows}
              offset={(page - 1) * PAGE_SIZE}
              selected={selected?.work_index ?? null}
              isLoading={scores.isFetching}
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
            explain={explain.data?.data ?? null}
            isLoading={explain.isFetching}
            isMock={explain.data?.source === "mock"}
            onLabel={onLabel}
          />
        </aside>

        <div className="xl:col-span-2">
          <NetworkGraph
            graph={graph.data?.data ?? null}
            isLoading={graph.isLoading}
            isMock={graph.data?.source === "mock"}
            selectedWorkIndex={selected?.work_index ?? null}
            onSelectEdge={(workIndex) => {
              const row = rows.find((r) => r.work_index === workIndex);
              if (row) setSelected(row);
            }}
          />
        </div>
      </main>
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
