import { formatINR, type LabelValue, type RiskRow, type ScoringMethod } from "@/lib/api";
import { cn } from "@/lib/utils";

function LabelBadge({ value }: { value: string }) {
  const styles: Record<string, string> = {
    "Training Label": "border-flag-training/40 bg-flag-training/10 text-flag-training",
    "GAT-Top": "border-flag-top/40 bg-flag-top/10 text-flag-top",
    New: "border-border bg-muted text-muted-foreground",
  };
  return (
    <span
      className={cn(
        "inline-block whitespace-nowrap border px-1.5 py-0.5 text-[11px] font-medium tracking-wide uppercase",
        styles[value] ?? "border-border bg-muted text-muted-foreground",
      )}
    >
      {value}
    </span>
  );
}

function BandBadge({ band }: { band: "Top 1%" | "Top 5%" | "Rest" }) {
  const styles: Record<string, string> = {
    "Top 1%": "border-foreground/35 bg-secondary text-foreground font-semibold",
    "Top 5%": "border-border bg-secondary/70 text-foreground/80 font-medium",
    Rest: "border-border/60 bg-muted/40 text-muted-foreground font-normal",
  };
  return (
    <span
      className={cn(
        "inline-block whitespace-nowrap border px-1.5 py-0.5 text-[11px] tracking-wide uppercase",
        styles[band],
      )}
    >
      {band}
    </span>
  );
}

interface Props {
  rows: RiskRow[];
  offset: number;
  total?: number;
  selected: number | null;
  isLoading: boolean;
  isDemo?: boolean;
  labellingEnabled?: boolean;
  method?: ScoringMethod;
  direction?: "higher_is_riskier" | "lower_is_riskier" | undefined;
  rowFeedback?: Record<number, { status: "success" | "duplicate" | "error"; message: string }>;
  onSelect: (row: RiskRow) => void;
  onLabel: (workIndex: number, label: LabelValue) => void;
}

export function RiskTable({
  rows,
  offset,
  total = 15000,
  selected,
  isLoading,
  isDemo = false,
  labellingEnabled = false,
  method = "gat",
  direction,
  rowFeedback,
  onSelect,
  onLabel,
}: Props) {
  const denom = 15000;
  const scoreHeader =
    method === "gat"
      ? "GAT score"
      : method === "siamese"
      ? "Siamese score"
      : method === "community"
      ? "Community score"
      : method === "fusion"
      ? "Fusion score"
      : "Triage Score";

  const headers = [
    "Rank",
    "MP",
    "Implementing Agency",
    "Amount (₹)",
    "Lag (days)",
    "Status",
    scoreHeader,
    "Percentile",
    "Band",
    "Label Status",
    ...(labellingEnabled ? ["Mark"] : []),
  ];

  const scoreHeaderTooltip =
    method === "siamese"
      ? "Lower distance = more similar to labelled cases. Not proof of misconduct."
      : method === "community"
      ? "Several MPs sanctioning similar amounts at one agency within 30 days. Not proof of misconduct."
      : method === "fusion"
      ? "Experimental rank-ordering score combining detector signals. Not proof of misconduct."
      : "A high score means the work is statistically unusual compared with other works. It is not proof of misconduct.";

  const getScoreCellTooltip = (score: number) => {
    const raw = score.toFixed(3);
    if (method === "siamese") {
      return `Raw distance: ${raw}. Lower distance = more similar to labelled cases. Not proof of misconduct.`;
    }
    if (method === "community") {
      return `Raw score: ${raw}. Several MPs sanctioning similar amounts at one agency within 30 days. Not proof of misconduct.`;
    }
    if (method === "fusion") {
      return `Raw score: ${raw}. Experimental rank-ordering score combining detector signals. Not proof of misconduct.`;
    }
    return `Raw score: ${raw}. A high score means the work is statistically unusual compared with other works. It is not proof of misconduct.`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-border bg-secondary/70 text-left">
            {headers.map((h) => {
              const isScoreCol = h === scoreHeader;
              return (
                <th
                  key={h}
                  className={cn(
                    "px-3 py-2 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase",
                    (h === "Amount (₹)" ||
                      h === "Lag (days)" ||
                      isScoreCol ||
                      h === "Percentile") &&
                      "text-right",
                  )}
                >
                  {h === "Rank" ? (
                    <div>
                      <span>Rank</span>
                      <span className="block text-[9px] font-normal normal-case tracking-normal text-muted-foreground whitespace-nowrap">
                        Ranks are global (all 15,000 works)
                      </span>
                    </div>
                  ) : isScoreCol ? (
                    <span
                      title={scoreHeaderTooltip}
                      className="cursor-help underline decoration-dotted underline-offset-2"
                    >
                      {h}
                    </span>
                  ) : (
                    h
                  )}
                  {isScoreCol && direction === "lower_is_riskier" && (
                    <span className="block text-[9px] font-normal lowercase tracking-normal text-muted-foreground">
                      (lower distance = more similar)
                    </span>
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody className={cn(isLoading && "opacity-50")}>
          {rows.map((row, i) => {
            const isSelected = selected === row.work_index;
            const feedback = rowFeedback?.[row.work_index];
            const rank = row.global_rank ?? (offset + i + 1);
            const pct = (rank / denom) * 100;
            const percentileStr = `Top ${pct.toFixed(2)}%`;
            const band: "Top 1%" | "Top 5%" | "Rest" =
              pct <= 1.0 ? "Top 1%" : pct <= 5.0 ? "Top 5%" : "Rest";

            return (
              <tr
                key={row.work_index}
                onClick={() => onSelect(row)}
                className={cn(
                  "cursor-pointer border-b border-border/70 transition-colors hover:bg-accent/40",
                  isSelected && "bg-accent/60",
                )}
              >
                <td className="px-3 py-2 font-mono text-muted-foreground tabular-nums">
                  {rank}
                </td>
                <td className="px-3 py-2 font-medium whitespace-nowrap">{row.MP}</td>
                <td className="max-w-[190px] truncate px-3 py-2 text-muted-foreground">{row.IDA}</td>
                <td className="px-3 py-2 text-right font-mono tabular-nums">
                  {formatINR(row.amount)}
                </td>
                <td
                  className={cn(
                    "px-3 py-2 text-right font-mono tabular-nums",
                    row.sanction_lag_days > 365 && "text-alert",
                  )}
                >
                  {row.sanction_lag_days}
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-muted-foreground">{row.status}</td>
                <td
                  className="px-3 py-2 text-right font-mono tabular-nums cursor-help"
                  title={getScoreCellTooltip(row.score)}
                >
                  {row.score.toFixed(3)}
                </td>
                <td
                  className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground"
                  title={`Rank ${rank} of ${denom.toLocaleString("en-IN")}`}
                >
                  {percentileStr}
                </td>
                <td className="px-3 py-2">
                  <BandBadge band={band} />
                </td>
                <td className="px-3 py-2">
                  <LabelBadge value={row.label_status} />
                </td>
                {labellingEnabled && (
                  <td className="px-3 py-2 whitespace-nowrap">
                    <div className="flex items-center gap-1.5">
                      <button
                        disabled={isDemo}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (isDemo) return;
                          onLabel(row.work_index, "s");
                        }}
                        title="Flag for review"
                        className={cn(
                          "w-6 border border-alert/40 py-0.5 text-[11px] font-medium text-alert hover:bg-alert/10 cursor-pointer",
                          isDemo && "opacity-50 cursor-not-allowed hover:bg-transparent",
                        )}
                      >
                        S
                      </button>
                      <button
                        disabled={isDemo}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (isDemo) return;
                          onLabel(row.work_index, "n");
                        }}
                        title="Mark normal"
                        className={cn(
                          "w-6 border border-ok/40 py-0.5 text-[11px] font-medium text-ok hover:bg-ok/10 cursor-pointer",
                          isDemo && "opacity-50 cursor-not-allowed hover:bg-transparent",
                        )}
                      >
                        N
                      </button>
                      {isDemo ? (
                        <span className="ml-1 text-[11px] text-muted-foreground italic">
                          Labelling disabled in demo mode.
                        </span>
                      ) : (
                        feedback && (
                          <span
                            className={cn(
                              "ml-1 text-[11px] font-medium",
                              feedback.status === "duplicate"
                                ? "text-amber-500 font-semibold"
                                : feedback.status === "success"
                                ? "text-ok"
                                : "text-destructive"
                            )}
                          >
                            {feedback.message}
                          </span>
                        )
                      )}
                    </div>
                  </td>
                )}
              </tr>
            );
          })}
          {rows.length === 0 && !isLoading && (
            <tr>
              <td colSpan={labellingEnabled ? 11 : 10} className="px-3 py-10 text-center text-muted-foreground">
                No records returned.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
