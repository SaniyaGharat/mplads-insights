import { formatINR, type ExplainResponse, type LabelValue, type RiskRow } from "@/lib/api";

interface Props {
  row: RiskRow | null;
  explain: ExplainResponse | null;
  isLoading: boolean;
  isMock: boolean;
  onLabel: (workIndex: number, label: LabelValue) => void;
}

export function ExplainPanel({ row, explain, isLoading, isMock, onLabel }: Props) {
  if (!row) {
    return (
      <div className="border border-border bg-card p-5 text-sm text-muted-foreground">
        Select a row in the risk-ranked list to view its explainability breakdown.
      </div>
    );
  }

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
          ["Anomaly score", row.score.toFixed(3)],
        ].map(([k, v]) => (
          <div key={k}>
            <dt className="text-[11px] tracking-wider text-muted-foreground uppercase">{k}</dt>
            <dd className="mt-0.5 font-mono tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>

      <div className="px-5 py-4">
        <h3 className="text-[11px] tracking-wider text-muted-foreground uppercase">
          Top 3 influencing neighbours
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

      <div className="flex flex-wrap gap-2 border-t border-border bg-secondary/50 px-5 py-3">
        <button
          onClick={() => onLabel(row.work_index, "s")}
          className="border border-alert/50 bg-alert/10 px-3 py-1.5 text-xs font-medium text-alert hover:bg-alert/20"
        >
          Mark Suspicious
        </button>
        <button
          onClick={() => onLabel(row.work_index, "n")}
          className="border border-ok/50 bg-ok/10 px-3 py-1.5 text-xs font-medium text-ok hover:bg-ok/20"
        >
          Mark Normal
        </button>
        <button
          onClick={() => onLabel(row.work_index, "skip")}
          className="border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-muted"
        >
          Skip
        </button>
      </div>
    </div>
  );
}
