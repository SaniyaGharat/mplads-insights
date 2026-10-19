import { useMemo } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  ComposedChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import {
  formatINR,
  type OverviewResponse,
  type ScoringMethod,
  type MethodInfo,
  DEFAULT_METHODS,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  data: OverviewResponse | null;
  isLoading: boolean;
  isMock: boolean;
  error?: string | undefined;
  method: ScoringMethod;
  setMethod: (m: ScoringMethod) => void;
  flagPct: 1 | 5 | 10;
  setFlagPct: (pct: 1 | 5 | 10) => void;
  methodList?: MethodInfo[] | undefined;
}

const SAGE = "oklch(0.6 0.05 155)";
const NAVY = "oklch(0.36 0.055 255)";
const ALERT = "oklch(0.65 0.22 25)";
const MUTED = "oklch(0.6 0.02 255)";
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
    <div className={cn("border border-border bg-card shadow-xs", className)}>
      <div className="border-b border-border px-5 py-3.5">
        <h3 className="text-sm font-semibold tracking-tight">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function KpiCard({
  label,
  value,
  subvalue,
  accent,
}: {
  label: string;
  value: string;
  subvalue?: string;
  accent?: string;
}) {
  return (
    <div className="border border-border bg-card px-5 py-4 shadow-xs">
      <p className="text-[11px] font-medium tracking-[0.12em] text-muted-foreground uppercase">{label}</p>
      <p className={cn("mt-1.5 font-mono text-2xl font-semibold tabular-nums", accent ?? "text-foreground")}>
        {value}
      </p>
      {subvalue && <p className="mt-1 text-xs text-muted-foreground">{subvalue}</p>}
    </div>
  );
}

export function OverviewTab({
  data,
  isLoading,
  isMock,
  error,
  method,
  setMethod,
  flagPct,
  setFlagPct,
  methodList = DEFAULT_METHODS,
}: Props) {
  const kpis = data?.kpis;

  const currentMethodInfo =
    methodList.find((m) => m.id === method) ?? DEFAULT_METHODS.find((m) => m.id === method);

  // Split scatter points into flagged vs normal for distinct rendering layers
  const { flaggedPoints, normalPoints } = useMemo(() => {
    if (!data?.scatter_points) return { flaggedPoints: [], normalPoints: [] };
    const flagged: typeof data.scatter_points = [];
    const normal: typeof data.scatter_points = [];
    for (const p of data.scatter_points) {
      if (p.flagged) flagged.push(p);
      else normal.push(p);
    }
    return { flaggedPoints: flagged, normalPoints: normal };
  }, [data?.scatter_points]);

  // Top 15 states for clean chart display
  const topStates = useMemo(() => {
    if (!data?.by_state) return [];
    return data.by_state.slice(0, 15);
  }, [data?.by_state]);

  return (
    <div className="space-y-6">
      {/* Required Audit Notice Banner */}
      <div className="flex items-center justify-between gap-4 border border-primary/20 bg-primary/5 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-2 w-2 rounded-full bg-primary" />
          <p className="text-xs font-semibold text-foreground">
            Scores prioritise audit attention; they are not findings of misconduct
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>Active filter scope applies across all visualisations</span>
        </div>
      </div>

      {/* Control Bar: Method & Cutoff Selectors */}
      <div className="flex flex-wrap items-center justify-between gap-4 border border-border bg-card p-4">
        {/* Scoring Method Selector */}
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
            Scoring Method
          </span>
          <div className="flex flex-wrap border border-border">
            {methodList.map((m) => {
              const isSelected = method === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setMethod(m.id)}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium cursor-pointer transition-colors",
                    isSelected
                      ? "bg-primary text-primary-foreground font-semibold"
                      : "bg-card text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
                  )}
                >
                  {m.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Review Flag Cutoff Selector */}
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
            Flag for Review Cutoff
          </span>
          <div className="flex border border-border">
            {([1, 5, 10] as const).map((pct) => (
              <button
                key={pct}
                onClick={() => setFlagPct(pct)}
                className={cn(
                  "px-3.5 py-1.5 text-xs font-mono font-medium cursor-pointer transition-colors",
                  flagPct === pct
                    ? "bg-alert text-white font-semibold"
                    : "bg-card text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
                )}
              >
                Top {pct}%
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Method Note */}
      {currentMethodInfo && (
        <p className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">{currentMethodInfo.name}:</span>{" "}
          {currentMethodInfo.note}
        </p>
      )}

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <KpiCard
          label="Total Sanctioned Works"
          value={isLoading ? "…" : (kpis?.total_works ?? 15000).toLocaleString("en-IN")}
          subvalue="All evaluated works"
        />
        <KpiCard
          label="Total Sanction Amount"
          value={isLoading ? "…" : formatINR(kpis?.total_amount ?? 0)}
          subvalue="Across all IDA authorities"
        />
        <KpiCard
          label="Average Sanction Lag"
          value={isLoading ? "…" : `${Math.round(kpis?.avg_lag_days ?? 0)} days`}
          subvalue="Recommendation to sanction"
        />
        <KpiCard
          label={`Works Flagged for Review (${flagPct}%)`}
          value={isLoading ? "…" : (kpis?.flagged_count ?? 0).toLocaleString("en-IN")}
          subvalue={`Lead for review in top ${flagPct}% score position`}
          accent="text-alert font-bold"
        />
        <KpiCard
          label="Amount in Flagged Works"
          value={isLoading ? "…" : formatINR(kpis?.flagged_amount ?? 0)}
          subvalue={`${kpis?.flagged_amount_share ?? 0}% of total sanction budget`}
          accent="text-alert"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* 1. Flagged Works by State */}
        <ChartCard
          title="Flagged Works by State"
          subtitle={`Number of works flagged for review in top ${flagPct}% score position by state (top 15 states)`}
        >
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topStates} margin={{ top: 10, right: 20, left: 0, bottom: 45 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="state"
                  {...axisProps}
                  interval={0}
                  angle={-35}
                  textAnchor="end"
                  height={55}
                />
                <YAxis {...axisProps} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: `1px solid ${GRID}`,
                    borderRadius: 3,
                    fontSize: 12,
                    fontFamily: "var(--font-mono)",
                  }}
                  formatter={(val: number, name: string) => [
                    val.toLocaleString("en-IN"),
                    name === "flagged_count" ? "Flagged for review" : "Total works",
                  ]}
                  labelStyle={{ color: AXIS, fontWeight: 600 }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, paddingTop: 6 }}
                  formatter={(val: string) => (val === "flagged_count" ? "Flagged for review" : "Total works")}
                />
                <Bar dataKey="flagged_count" name="flagged_count" fill={ALERT} radius={[2, 2, 0, 0]} />
                <Bar dataKey="total_count" name="total_count" fill={NAVY} radius={[2, 2, 0, 0]} opacity={0.35} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* 2. Flagged Works by Status */}
        <ChartCard
          title="Flagged Works by Sanction Status"
          subtitle="Distribution of works flagged for review across work lifecycle stages"
        >
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data?.by_status ?? []}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 35, bottom: 10 }}
              >
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" {...axisProps} allowDecimals={false} />
                <YAxis
                  dataKey="status"
                  type="category"
                  {...axisProps}
                  width={110}
                  tick={{ fill: AXIS, fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: `1px solid ${GRID}`,
                    borderRadius: 3,
                    fontSize: 12,
                    fontFamily: "var(--font-mono)",
                  }}
                  formatter={(val: number, name: string) => [
                    val.toLocaleString("en-IN"),
                    name === "flagged_count" ? "Flagged for review" : "Total works",
                  ]}
                  labelStyle={{ color: AXIS, fontWeight: 600 }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, paddingTop: 6 }}
                  formatter={(val: string) => (val === "flagged_count" ? "Flagged for review" : "Total works")}
                />
                <Bar dataKey="flagged_count" name="flagged_count" fill={ALERT} radius={[0, 2, 2, 0]} />
                <Bar dataKey="total_count" name="total_count" fill={SAGE} radius={[0, 2, 2, 0]} opacity={0.35} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* 3. Monthly Sanction Timeline */}
        <ChartCard
          title="Monthly Sanction Timeline with Flagged Share"
          subtitle="Monthly sanction volume and proportion of works flagged for review"
        >
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={data?.monthly_timeline ?? []}
                margin={{ top: 10, right: 30, left: 10, bottom: 25 }}
              >
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="month"
                  {...axisProps}
                  tick={{ fill: AXIS, fontSize: 10 }}
                  interval={2}
                />
                <YAxis
                  yAxisId="left"
                  {...axisProps}
                  label={{
                    value: "Works Count",
                    angle: -90,
                    position: "insideLeft",
                    style: { fill: AXIS, fontSize: 10 },
                  }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  {...axisProps}
                  unit="%"
                  label={{
                    value: "Flagged Share (%)",
                    angle: 90,
                    position: "insideRight",
                    style: { fill: AXIS, fontSize: 10 },
                  }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: `1px solid ${GRID}`,
                    borderRadius: 3,
                    fontSize: 12,
                    fontFamily: "var(--font-mono)",
                  }}
                  formatter={(val: number, name: string) => [
                    name === "flagged_share" ? `${val}%` : val.toLocaleString("en-IN"),
                    name === "flagged_share"
                      ? "Flagged share"
                      : name === "flagged_count"
                        ? "Flagged for review"
                        : "Total sanctions",
                  ]}
                  labelStyle={{ color: AXIS, fontWeight: 600 }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, paddingTop: 6 }}
                  formatter={(val: string) =>
                    val === "flagged_share"
                      ? "Share flagged for review (%)"
                      : val === "flagged_count"
                        ? "Flagged for review count"
                        : "Total works"
                  }
                />
                <Bar
                  yAxisId="left"
                  dataKey="total_count"
                  name="total_count"
                  fill={NAVY}
                  opacity={0.35}
                  radius={[2, 2, 0, 0]}
                />
                <Bar
                  yAxisId="left"
                  dataKey="flagged_count"
                  name="flagged_count"
                  fill={ALERT}
                  opacity={0.8}
                  radius={[2, 2, 0, 0]}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="flagged_share"
                  name="flagged_share"
                  stroke={ALERT}
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* 4. Lag vs Amount Scatter Plot */}
        <ChartCard
          title="Sanction Lag vs Amount Distribution"
          subtitle="Scatter of works showing lag days vs sanction amount (sampled up to 3,000 works for rapid rendering)"
        >
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 30, left: 15, bottom: 20 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="lag"
                  name="Sanction Lag"
                  unit="d"
                  {...axisProps}
                  label={{
                    value: "Sanction Lag (Days)",
                    position: "insideBottom",
                    offset: -12,
                    style: { fill: AXIS, fontSize: 11 },
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="amount"
                  name="Amount"
                  {...axisProps}
                  tickFormatter={(val: number) => `₹${(val / 100000).toFixed(0)}L`}
                  label={{
                    value: "Sanction Amount (₹)",
                    angle: -90,
                    position: "insideLeft",
                    style: { fill: AXIS, fontSize: 11 },
                  }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ payload }) => {
                    const firstItem = payload && payload.length > 0 ? payload[0] : null;
                    if (!firstItem || !firstItem.payload) return null;
                    const p = firstItem.payload;
                    return (
                      <div
                        style={{
                          background: "var(--card)",
                          border: `1px solid ${GRID}`,
                          borderRadius: 3,
                          padding: "8px 12px",
                          fontSize: 12,
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        <p className="font-semibold text-foreground">
                          Work #{p.work_index} · Score Position #{p.rank}
                        </p>
                        <p className={p.flagged ? "text-alert font-bold" : "text-muted-foreground"}>
                          {p.flagged ? "Flagged for review (Lead for review)" : "Normal work"}
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          Amount: {formatINR(p.amount)}
                        </p>
                        <p className="text-xs text-muted-foreground">Lag: {p.lag} days</p>
                        <p className="text-xs text-muted-foreground">
                          {p.state} · {p.status}
                        </p>
                      </div>
                    );
                  }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, paddingTop: 6 }}
                  formatter={(val: string) =>
                    val === "flagged"
                      ? `Lead for review (Top ${flagPct}% score position)`
                      : "Normal work sample"
                  }
                />
                <Scatter
                  name="normal"
                  data={normalPoints}
                  fill={MUTED}
                  opacity={0.3}
                  shape="circle"
                />
                <Scatter
                  name="flagged"
                  data={flaggedPoints}
                  fill={ALERT}
                  opacity={0.9}
                  shape="circle"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground text-center">
            Points represent a sample of up to 3,000 works (all flagged works at {flagPct}% cutoff + balanced sample of normal works).
          </p>
        </ChartCard>
      </div>
    </div>
  );
}
