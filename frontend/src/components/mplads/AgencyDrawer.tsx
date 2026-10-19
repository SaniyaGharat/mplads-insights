import { useQuery } from "@tanstack/react-query";
import { X, Building2, AlertCircle } from "lucide-react";
import {
  fetchAgency,
  formatINR,
  type ScoringMethod,
  type AgencyResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  ida: string | null;
  onClose: () => void;
  method: ScoringMethod;
  flagPct: 1 | 5 | 10;
}

export function AgencyDrawer({ ida, onClose, method, flagPct }: Props) {
  const agencyQuery = useQuery({
    queryKey: ["agency", ida, method, flagPct],
    queryFn: () => fetchAgency(ida!, method, flagPct),
    enabled: Boolean(ida),
  });

  if (!ida) return null;

  const data: AgencyResponse | undefined = agencyQuery.data?.data;
  const isLoading = agencyQuery.isLoading;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-xs transition-opacity animate-in fade-in">
      {/* Backdrop click to close */}
      <div className="absolute inset-0" onClick={onClose} />

      {/* Drawer Panel */}
      <div className="relative z-10 flex h-full w-full max-w-3xl flex-col border-l border-border bg-card shadow-2xl animate-in slide-in-from-right duration-200">
        {/* Header */}
        <div className="flex items-start justify-between border-b border-border px-6 py-5">
          <div className="space-y-1 pr-6">
            <div className="flex items-center gap-2">
              <Building2 className="h-4 w-4 text-primary" />
              <p className="text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
                Implementing Agency Profile
              </p>
            </div>
            <h2 className="text-lg font-semibold tracking-tight text-foreground">
              {data?.ida ?? ida}
            </h2>
            {data && (
              <p className="text-xs text-muted-foreground">
                District: <span className="font-semibold text-foreground">{data.district}</span> · State:{" "}
                <span className="font-semibold text-foreground">{data.state}</span>
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="rounded-xs border border-border p-1.5 text-muted-foreground hover:bg-secondary hover:text-foreground cursor-pointer"
            aria-label="Close agency drawer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Audit Guidance Notice */}
        <div className="border-b border-primary/20 bg-primary/5 px-6 py-3">
          <p className="text-xs font-semibold text-foreground">
            Scores prioritise audit attention; they are not findings of misconduct
          </p>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
          {isLoading ? (
            <div className="flex h-64 items-center justify-center">
              <p className="text-sm text-muted-foreground">Loading agency data…</p>
            </div>
          ) : !data ? (
            <div className="flex h-64 flex-col items-center justify-center gap-2 text-center">
              <AlertCircle className="h-6 w-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Agency records could not be loaded.</p>
            </div>
          ) : (
            <>
              {/* Summary KPIs */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                <div className="border border-border bg-card/60 p-3.5">
                  <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">Works</p>
                  <p className="mt-1 font-mono text-xl font-semibold">{data.works.toLocaleString("en-IN")}</p>
                  <p className="text-[11px] text-muted-foreground">
                    {data.is_small_group ? "Small agency" : "All sanctions"}
                  </p>
                </div>
                <div className="border border-border bg-card/60 p-3.5">
                  <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">Total Amount</p>
                  <p className="mt-1 font-mono text-xl font-semibold">{formatINR(data.total_amount)}</p>
                  <p className="text-[11px] text-muted-foreground">Sanctioned</p>
                </div>
                <div className="border border-border bg-card/60 p-3.5">
                  <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">Median Amount</p>
                  <p className="mt-1 font-mono text-xl font-semibold text-foreground">{formatINR(data.median_amount)}</p>
                  <p className="text-[11px] text-muted-foreground">Agency baseline</p>
                </div>
                <div className="border border-border bg-card/60 p-3.5">
                  <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">Median Lag</p>
                  <p className="mt-1 font-mono text-xl font-semibold text-foreground">{Math.round(data.median_lag)} d</p>
                  <p className="text-[11px] text-muted-foreground">Agency baseline</p>
                </div>
                <div className="border border-border bg-card/60 p-3.5">
                  <p className="text-[10px] font-medium tracking-wider text-muted-foreground uppercase">Flag for Review</p>
                  <p className="mt-1 font-mono text-xl font-semibold text-alert">{data.flagged_count}</p>
                  <p className="text-[11px] text-muted-foreground">
                    {data.flagged_share}% (Top {flagPct}%)
                  </p>
                </div>
              </div>

              {/* Status Mix */}
              <div className="border border-border bg-card p-4">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Work Status Mix
                </h4>
                <div className="mt-3 flex flex-wrap gap-2">
                  {data.status_mix.map((item) => (
                    <div
                      key={item.status}
                      className="flex items-center gap-2 border border-border bg-secondary/40 px-3 py-1.5 text-xs"
                    >
                      <span className="font-medium text-foreground">{item.status}</span>
                      <span className="font-mono text-muted-foreground">
                        {item.count} ({item.share}%)
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Agency Works Table */}
              <div className="space-y-2">
                <div className="flex items-baseline justify-between">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Agency Works Table ({data.works_data.length})
                  </h4>
                  <p className="text-[11px] text-muted-foreground">
                    Sorted by score position · Showing agency baseline alongside each work
                  </p>
                </div>

                <div className="overflow-x-auto border border-border">
                  <table className="w-full text-left text-xs">
                    <thead className="border-b border-border bg-secondary/50 font-medium text-muted-foreground">
                      <tr>
                        <th className="px-3 py-2.5">Score Position</th>
                        <th className="px-3 py-2.5">Work #</th>
                        <th className="px-3 py-2.5">MP</th>
                        <th className="px-3 py-2.5">Category</th>
                        <th className="px-3 py-2.5 text-right">Sanction Amount</th>
                        <th className="px-3 py-2.5 text-right">Agency Median Amount</th>
                        <th className="px-3 py-2.5 text-right">Lag (Days)</th>
                        <th className="px-3 py-2.5 text-right">Agency Median Lag</th>
                        <th className="px-3 py-2.5">Status</th>
                        <th className="px-3 py-2.5 text-center">Review Lead</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {data.works_data.map((row) => (
                        <tr
                          key={row.work_index}
                          className={cn(
                            "transition-colors hover:bg-secondary/30",
                            row.review_flag && "bg-alert/5",
                          )}
                        >
                          <td className="px-3 py-2 font-mono font-medium">#{row.global_rank}</td>
                          <td className="px-3 py-2 font-mono text-muted-foreground">#{row.work_index}</td>
                          <td className="px-3 py-2 font-medium text-foreground">{row.MP}</td>
                          <td className="px-3 py-2 text-muted-foreground">{row.category}</td>
                          <td className="px-3 py-2 text-right font-mono font-medium text-foreground">
                            {formatINR(row.amount)}
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-muted-foreground">
                            {formatINR(row.agency_median_amount)}
                          </td>
                          <td className="px-3 py-2 text-right font-mono font-medium text-foreground">
                            {row.sanction_lag_days} d
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-muted-foreground">
                            {Math.round(row.agency_median_lag)} d
                          </td>
                          <td className="px-3 py-2 text-muted-foreground">{row.status}</td>
                          <td className="px-3 py-2 text-center">
                            {row.review_flag ? (
                              <span className="inline-block border border-alert/30 bg-alert/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-alert">
                                Flag for review
                              </span>
                            ) : (
                              <span className="inline-block px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                                Normal
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Neutrality Note */}
              <div className="border-t border-border pt-4 text-center">
                <p className="text-[11px] text-muted-foreground">
                  Scores prioritise audit attention; they are not findings of misconduct. No agency risk score and no agency ranking by flagged share.
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
