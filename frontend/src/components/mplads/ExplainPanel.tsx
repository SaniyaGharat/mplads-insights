import { useQuery } from "@tanstack/react-query";
import {
  formatINR,
  fetchWorkCompare,
  type ExplainResponse,
  type LabelValue,
  type RiskRow,
  type ScoringMethod,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  row: RiskRow | null;
  method?: ScoringMethod;
  explain: ExplainResponse | null;
  isLoading: boolean;
  isMock: boolean;
  isDemo?: boolean;
  labellingEnabled?: boolean;
  flagPct?: 1 | 5 | 10;
  feedback?: { status: "success" | "duplicate" | "error"; message: string } | null | undefined;
  onLabel: (workIndex: number, label: LabelValue) => void;
}

const METHOD_LABELS: Record<string, { name: string; tag?: string }> = {
  gat: { name: "GAT" },
  siamese: { name: "Siamese" },
  community: { name: "Community" },
  fusion: { name: "Fusion", tag: "Experimental" },
};

export function ExplainPanel({
  row,
  method = "gat",
  explain,
  isLoading,
  isMock,
  isDemo = false,
  labellingEnabled = false,
  flagPct = 5,
  feedback,
  onLabel,
}: Props) {
  const compareQuery = useQuery({
    queryKey: ["work-compare", row?.work_index],
    queryFn: () => fetchWorkCompare(row!.work_index),
    enabled: row !== null,
  });

  if (!row) {
    return (
      <div className="border border-border bg-card p-5 text-sm text-muted-foreground">
        Select a row in the risk-ranked list to view its explainability breakdown.
      </div>
    );
  }

  const compareData = compareQuery.data?.data;
  const methodsOrder: ScoringMethod[] = ["gat", "siamese", "community", "fusion"];

  return (
    <div className="border border-border bg-card">
      <div className="border-b border-border px-5 py-3">
        <h2 className="text-base font-semibold">Explainability — work #{row.work_index}</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          {row.MP} · {row.IDA} · {row.category}
        </p>
      </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 border-b border-border px-5 py-4 text-[13px] sm:grid-cols-4">
        {[
          ["Amount", formatINR(row.amount)],
          ["Sanction lag", `${row.sanction_lag_days} days`],
          ["Status", row.status],
          ["Triage score", row.score.toFixed(3)],
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-[11px] tracking-wider text-muted-foreground uppercase">{k}</dt>
            <dd className="mt-0.5 font-mono tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>

      {/* Compare Methods Panel */}
      <div className="border-b border-border px-5 py-4">
        <div className="flex items-center justify-between">
          <h3 className="text-[11px] tracking-wider text-muted-foreground uppercase">
            Method Comparison (All 4 Methods)
          </h3>
          <span className="text-[10px] text-muted-foreground">
            Cutoff: Top {flagPct}%
          </span>
        </div>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Score position and global ranking across models (15,000 total works).
        </p>

        {compareQuery.isLoading ? (
          <p className="mt-3 text-xs text-muted-foreground">Loading method comparison…</p>
        ) : compareData?.methods ? (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs border border-border">
              <thead>
                <tr className="border-b border-border bg-secondary/40 text-[11px] text-muted-foreground">
                  <th className="px-2.5 py-1.5 font-medium">Method</th>
                  <th className="px-2.5 py-1.5 font-medium text-right">Global Rank</th>
                  <th className="px-2.5 py-1.5 font-medium text-right">Percentile</th>
                  <th className="px-2.5 py-1.5 font-medium text-right">Score</th>
                  <th className="px-2.5 py-1.5 font-medium text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {methodsOrder.map((mKey) => {
                  const mInfo = compareData.methods[mKey];
                  const meta = METHOD_LABELS[mKey] ?? { name: mKey };
                  const isCurrent = method === mKey;
                  const isFlagged = mInfo ? mInfo.percentile <= flagPct : false;

                  return (
                    <tr
                      key={mKey}
                      className={cn(
                        "transition-colors",
                        isCurrent ? "bg-primary/5 font-medium" : "hover:bg-muted/40",
                      )}
                    >
                      <td className="px-2.5 py-2">
                        <div className="flex items-center gap-1.5">
                          <span className={cn(isCurrent && "font-semibold text-foreground")}>
                            {meta.name}
                          </span>
                          {meta.tag && (
                            <span className="rounded bg-amber-500/15 px-1 py-0.2 text-[9px] font-semibold text-amber-700 dark:text-amber-400">
                              {meta.tag}
                            </span>
                          )}
                          {mKey === "siamese" && (
                            <span
                              title="Siamese was trained on the 51 analyst labels, so those works appear flagged by construction."
                              className="cursor-help rounded bg-secondary px-1 text-[9px] text-muted-foreground"
                            >
                              trained on 51 labels ℹ️
                            </span>
                          )}
                          {isCurrent && (
                            <span className="rounded bg-secondary px-1 text-[9px] text-muted-foreground">
                              active
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono tabular-nums">
                        {mInfo ? `#${mInfo.global_rank.toLocaleString("en-IN")}` : "—"}
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono tabular-nums">
                        {mInfo ? `Top ${mInfo.percentile.toFixed(1)}%` : "—"}
                      </td>
                      <td className="px-2.5 py-2 text-right font-mono tabular-nums text-muted-foreground">
                        {mInfo ? mInfo.score.toFixed(3) : "—"}
                      </td>
                      <td className="px-2.5 py-2 text-center">
                        <span
                          className={cn(
                            "inline-block rounded px-1.5 py-0.5 text-[10px] font-medium leading-tight",
                            isFlagged
                              ? "bg-amber-500/15 text-amber-800 dark:text-amber-300"
                              : "bg-muted text-muted-foreground",
                          )}
                        >
                          {isFlagged ? "Flag for review" : "Not flagged"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="mt-2.5 rounded bg-secondary/30 p-2 text-[11px] text-muted-foreground space-y-1 border border-border/50">
              <p>
                • <strong>Method Disagreement:</strong> A work flagged by one method and "Not flagged" by another is the expected finding due to differing model inductive biases, not a bug.
              </p>
              <p>
                • <strong>Siamese Flags:</strong> Siamese was trained on the 51 analyst labels, so those works appear flagged by construction.
              </p>
            </div>
          </div>
        ) : (
          <p className="mt-3 text-xs text-muted-foreground italic">
            Comparison data unavailable.
          </p>
        )}
      </div>

      {method === "gat" ? (
        <div className="px-5 py-4">
          <h3 className="text-[11px] tracking-wider text-muted-foreground uppercase">
            Top 3 attention edges (neighbours may repeat)
          </h3>
          {isLoading && <p className="mt-3 text-sm text-muted-foreground">Loading attribution…</p>}
          {!isLoading && explain && (
            <ol className="mt-3 space-y-3">
              {explain.top_neighbors.slice(0, 3).map((n, i) => (
                <li key={`${n.neighbor}-${i}`}>
                  <div className="flex items-baseline justify-between gap-4 text-[13px]">
                    <span>
                      <span className="mr-2 font-mono text-muted-foreground">{i + 1}.</span>
                      {n.neighbor}
                    </span>
                    <span className="font-mono tabular-nums">{n.weight.toFixed(2)}</span>
                  </div>
                  <div className="mt-1 h-2 w-full bg-secondary">
                    <div
                      className="h-2 bg-sage"
                      style={{ width: `${Math.min(100, Math.max(2, n.weight * 100))}%` }}
                    />
                  </div>
                </li>
              ))}
            </ol>
          )}
          {isMock && !isLoading && (
            <p className="mt-3 text-xs text-muted-foreground">
              Backend unreachable — showing fallback attribution.
            </p>
          )}
        </div>
      ) : (
        <div className="px-5 py-6">
          <p className="text-sm text-muted-foreground">
            No per-work explanation available for this method
          </p>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-secondary/50 px-5 py-3">
        {labellingEnabled ? (
          <div className="flex flex-wrap items-center gap-2">
            <button
              disabled={isDemo}
              onClick={() => {
                if (isDemo) return;
                onLabel(row.work_index, "s");
              }}
              className={cn(
                "border border-alert/50 bg-alert/10 px-3 py-1.5 text-xs font-medium text-alert hover:bg-alert/20 cursor-pointer",
                isDemo && "opacity-50 cursor-not-allowed hover:bg-alert/10",
              )}
            >
              Flag for Review
            </button>
            <button
              disabled={isDemo}
              onClick={() => {
                if (isDemo) return;
                onLabel(row.work_index, "n");
              }}
              className={cn(
                "border border-ok/50 bg-ok/10 px-3 py-1.5 text-xs font-medium text-ok hover:bg-ok/20 cursor-pointer",
                isDemo && "opacity-50 cursor-not-allowed hover:bg-ok/10",
              )}
            >
              Mark Normal
            </button>
            <button
              disabled={isDemo}
              onClick={() => {
                if (isDemo) return;
                onLabel(row.work_index, "skip");
              }}
              className={cn(
                "border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-muted cursor-pointer",
                isDemo && "opacity-50 cursor-not-allowed hover:bg-transparent",
              )}
            >
              Skip
            </button>
            {isDemo && (
              <span className="text-xs text-muted-foreground italic">
                Labelling disabled in demo mode.
              </span>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground italic">
            Labels shown are the 60 analyst labels collected offline; labelling is disabled in this deployment.
          </p>
        )}
        {feedback && (
          <span
            className={cn(
              "text-xs font-medium tracking-wide",
              feedback.status === "duplicate"
                ? "text-amber-500 font-semibold"
                : feedback.status === "success"
                ? "text-ok"
                : "text-destructive"
            )}
          >
            {feedback.message}
          </span>
        )}
      </div>
    </div>
  );
}
