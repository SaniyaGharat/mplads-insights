import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatINR, type StatsResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  stats: StatsResponse | null;
  isLoading: boolean;
  isMock: boolean;
  error?: string | undefined;
}

const SAGE = "oklch(0.6 0.05 155)";
const NAVY = "oklch(0.36 0.055 255)";
const CLAY = "oklch(0.53 0.08 55)";
const GRID = "oklch(0.885 0.008 90)";
const AXIS = "oklch(0.5 0.02 255)";

const axisProps = {
  stroke: GRID,
  tick: { fill: AXIS, fontSize: 11 },
  tickLine: false,
} as const;

function ChartCard({
  title,
  subtitle,
  children,
  className,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("border border-border bg-card", className)}>
      <div className="border-b border-border px-5 py-3">
        <h3 className="text-sm font-semibold">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>}
      </div>
      <div className="px-2 py-4 pr-4">{children}</div>
    </div>
  );
}

function tooltipStyle() {
  return {
    contentStyle: {
      background: "var(--card)",
      border: `1px solid ${GRID}`,
      borderRadius: 2,
      fontSize: 12,
      fontFamily: "var(--font-mono)",
    },
    labelStyle: { color: AXIS },
  };
}

function Kpi({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: string;
}) {
  return (
    <div className="border border-border bg-card px-5 py-4">
      <p className="text-[11px] tracking-[0.14em] text-muted-foreground uppercase">{label}</p>
      <p className={cn("mt-1.5 font-mono text-2xl tabular-nums", accent)}>{value}</p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export function StatsOverview({ stats, isLoading, isMock, error }: Props) {
  if (isLoading || !stats) {
    return (
      <section className="border border-border bg-card px-5 py-12">
        <p className="text-sm text-muted-foreground">Loading dashboard summary…</p>
      </section>
    );
  }

  const topMps = [...stats.top_risky_mps]
    .sort((a, b) => b.flagged_count - a.flagged_count)
    .slice(0, 10);

  const lagData = [
    { name: "Flagged works", lag: stats.avg_lag_flagged_vs_normal.flagged_avg_lag },
    { name: "Normal works", lag: stats.avg_lag_flagged_vs_normal.normal_avg_lag },
  ];

  return (
    <section className="grid gap-5">
      {isMock && (
        <p className="border border-alert/30 bg-alert/8 px-5 py-2 text-xs text-alert">
          Summary endpoint unreachable ({error ?? "network error"}). Showing sample aggregates.
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <Kpi label="Total works reviewed" value={stats.total_works.toLocaleString("en-IN")} />
        <Kpi label="Total amount flagged" value={formatINR(stats.total_amount)} />
        <Kpi
          label="Avg. sanction lag"
          value={`${Math.round(stats.avg_lag_days).toLocaleString("en-IN")} d`}
        />
        <Kpi
          label="Training label"
          value={stats.label_counts.training_label.toLocaleString("en-IN")}
          accent="text-flag-training"
        />
        <Kpi
          label="GAT-Top"
          value={stats.label_counts.gat_top.toLocaleString("en-IN")}
          accent="text-flag-top"
        />
        <Kpi
          label="New"
          value={stats.label_counts.new.toLocaleString("en-IN")}
          accent="text-flag-new"
        />
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <ChartCard title="Works by status" subtitle="Count of sanctioned works per status">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={stats.status_breakdown}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="status" {...axisProps} />
              <YAxis {...axisProps} />
              <Tooltip {...tooltipStyle()} cursor={{ fill: "oklch(0.6 0.05 155 / 0.08)" }} />
              <Bar dataKey="count" fill={NAVY} maxBarSize={44} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Anomaly score distribution"
          subtitle="Equal-count buckets — note how wide the final bucket is compared to the rest"
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={stats.score_distribution} barCategoryGap={1}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis
                dataKey="bucket"
                {...axisProps}
                interval={0}
                angle={-35}
                height={64}
                dy={12}
                textAnchor="end"
                tick={{ fill: AXIS, fontSize: 10 }}
              />
              <YAxis {...axisProps} />
              <Tooltip {...tooltipStyle()} cursor={{ fill: "oklch(0.6 0.05 155 / 0.08)" }} />
              <Bar dataKey="count">
                {stats.score_distribution.map((d, i) => (
                  <Cell
                    key={d.bucket}
                    fill={i >= stats.score_distribution.length - 3 ? CLAY : SAGE}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-2 px-2 text-[11px] italic text-muted-foreground">
            Each bar holds roughly the same number of works (~10%). The skew is in the bucket
            widths, not the heights — the final bucket alone spans scores up to 13,913.
          </p>
        </ChartCard>

        <ChartCard title="Top 10 highest-risk MPs" subtitle="Ranked by flagged work count">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={topMps} layout="vertical" margin={{ left: 8 }}>
              <CartesianGrid stroke={GRID} horizontal={false} />
              <XAxis type="number" {...axisProps} />
              <YAxis type="category" dataKey="MP" width={150} {...axisProps} />
              <Tooltip
                {...tooltipStyle()}
                cursor={{ fill: "oklch(0.6 0.05 155 / 0.08)" }}
                formatter={(v: number, _n, item: { payload?: { total_flagged_amount: number } }) => [
                  `${v} flagged · ${formatINR(item.payload?.total_flagged_amount ?? 0)}`,
                  "Flagged",
                ]}
              />
              <Bar dataKey="flagged_count" fill={SAGE} maxBarSize={18} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Avg sanction lag: flagged vs normal works"
          subtitle="Days between sanction and recorded progress"
        >
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={lagData}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="name" {...axisProps} />
              <YAxis {...axisProps} unit="d" />
              <Tooltip {...tooltipStyle()} cursor={{ fill: "oklch(0.6 0.05 155 / 0.08)" }} />
              <Bar dataKey="lag" maxBarSize={90}>
                <Cell fill={CLAY} />
                <Cell fill={NAVY} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </section>
  );
}
