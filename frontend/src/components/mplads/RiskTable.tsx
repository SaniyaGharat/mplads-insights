import { formatINR, type LabelValue, type RiskRow } from "@/lib/api";
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

interface Props {
  rows: RiskRow[];
  offset: number;
  selected: number | null;
  isLoading: boolean;
  onSelect: (row: RiskRow) => void;
  onLabel: (workIndex: number, label: LabelValue) => void;
}

export function RiskTable({ rows, offset, selected, isLoading, onSelect, onLabel }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-border bg-secondary/70 text-left">
            {[
              "Rank",
              "MP",
              "Implementing Agency",
              "Amount (₹)",
              "Lag (days)",
              "Status",
              "Score",
              "Label Status",
              "Mark",
            ].map((h) => (
              <th
                key={h}
                className="px-3 py-2 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className={cn(isLoading && "opacity-50")}>
          {rows.map((row, i) => {
            const isSelected = selected === row.work_index;
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
                  {offset + i + 1}
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
                <td className="px-3 py-2 text-right font-mono tabular-nums">
                  {row.score.toFixed(3)}
                </td>
                <td className="px-3 py-2">
                  <LabelBadge value={row.label_status} />
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <div className="flex gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onLabel(row.work_index, "s");
                      }}
                      title="Mark suspicious"
                      className="w-6 border border-alert/40 py-0.5 text-[11px] font-medium text-alert hover:bg-alert/10"
                    >
                      S
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onLabel(row.work_index, "n");
                      }}
                      title="Mark normal"
                      className="w-6 border border-ok/40 py-0.5 text-[11px] font-medium text-ok hover:bg-ok/10"
                    >
                      N
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
          {rows.length === 0 && !isLoading && (
            <tr>
              <td colSpan={9} className="px-3 py-10 text-center text-muted-foreground">
                No records returned.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
