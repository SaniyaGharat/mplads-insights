import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useMutation, keepPreviousData } from "@tanstack/react-query";
import { useState, useEffect, useMemo } from "react";
import {
  fetchExplain,
  fetchFilterOptions,
  fetchGraph,
  fetchHealth,
  fetchMethods,
  fetchRiskScores,
  fetchStats,
  fetchLabelTotals,
  fetchOverview,
  fetchGeography,
  postLabel,
  DEFAULT_METHODS,
  type LabelTotals,
  type LabelValue,
  type RiskRow,
  type ScoringMethod,
  type RiskScoresFilterParams,
  METHOD_THRESHOLDS,
} from "@/lib/api";
import { API_BASE_URL } from "@/lib/config";
import { cn } from "@/lib/utils";
import { RiskTable } from "@/components/mplads/RiskTable";
import { StatsOverview } from "@/components/mplads/StatsOverview";
import { ExplainPanel } from "@/components/mplads/ExplainPanel";
import { NetworkGraph } from "@/components/mplads/NetworkGraph";
import { EvaluationTab } from "@/components/mplads/EvaluationTab";
import { OverviewTab } from "@/components/mplads/OverviewTab";
import { GeographyTab } from "@/components/mplads/GeographyTab";
import { AgencyDrawer } from "@/components/mplads/AgencyDrawer";

const TITLE = "MPLADS Anomaly Detection";
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

function parseUrlParams() {
  if (typeof window === "undefined") return null;
  const sp = new URLSearchParams(window.location.search);
  const m = sp.get("method")?.toLowerCase();
  const validMethods: ScoringMethod[] = ["gat", "siamese", "community", "fusion"];
  const flagPctParam = Number(sp.get("flag_pct"));
  const flagPctVal: 1 | 5 | 10 = flagPctParam === 1 || flagPctParam === 10 ? flagPctParam : 5;
  const tabParam = sp.get("tab")?.toLowerCase();
  const tabVal: "overview" | "geography" | "triage" | "evaluation" =
    tabParam === "evaluation"
      ? "evaluation"
      : tabParam === "geography"
      ? "geography"
      : tabParam === "triage"
      ? "triage"
      : "overview";

  return {
    method: validMethods.includes(m as ScoringMethod) ? (m as ScoringMethod) : ("gat" as ScoringMethod),
    page: Math.max(1, parseInt(sp.get("page") || "1", 10) || 1),
    search: sp.get("search") || "",
    state: sp.get("state") || "",
    district: sp.get("district") || "",
    status: sp.get("status") || "",
    minAmount: sp.get("min_amount") || "",
    maxAmount: sp.get("max_amount") || "",
    minLag: sp.get("min_lag") || "",
    maxLag: sp.get("max_lag") || "",
    labelStatus: sp.get("label_status") || "",
    flagPct: flagPctVal,
    tab: tabVal,
  };
}

function Dashboard() {
  const [activeTab, setActiveTab] = useState<"overview" | "geography" | "triage" | "evaluation">("overview");
  const [selectedAgencyId, setSelectedAgencyId] = useState<string | null>(null);
  const [flagPct, setFlagPct] = useState<1 | 5 | 10>(5);
  const [method, setMethod] = useState<ScoringMethod>("gat");
  const [page, setPage] = useState<number>(1);
  const [searchInput, setSearchInput] = useState<string>("");
  const [debouncedSearch, setDebouncedSearch] = useState<string>("");
  const [selectedState, setSelectedState] = useState<string>("");
  const [selectedDistrict, setSelectedDistrict] = useState<string>("");
  const [selectedStatus, setSelectedStatus] = useState<string>("");
  const [minAmount, setMinAmount] = useState<string>("");
  const [maxAmount, setMaxAmount] = useState<string>("");
  const [minLag, setMinLag] = useState<string>("");
  const [maxLag, setMaxLag] = useState<string>("");
  const [labelStatus, setLabelStatus] = useState<string>("");
  const [isMounted, setIsMounted] = useState<boolean>(false);

  const [selected, setSelected] = useState<RiskRow | null>(null);
  const [totals, setTotals] = useState<LabelTotals>({ s: 0, n: 0, skip: 0 });
  const [rowFeedback, setRowFeedback] = useState<
    Record<number, { status: "success" | "duplicate" | "error"; message: string }>
  >({});

  // Parse URL search params once on client mount to prevent SSR hydration mismatch
  useEffect(() => {
    setIsMounted(true);
    const p = parseUrlParams();
    if (p) {
      if (p.method !== "gat") setMethod(p.method);
      if (p.page !== 1) setPage(p.page);
      if (p.search) {
        setSearchInput(p.search);
        setDebouncedSearch(p.search);
      }
      if (p.state) setSelectedState(p.state);
      if (p.district) setSelectedDistrict(p.district);
      if (p.status) setSelectedStatus(p.status);
      if (p.minAmount) setMinAmount(p.minAmount);
      if (p.maxAmount) setMaxAmount(p.maxAmount);
      if (p.minLag) setMinLag(p.minLag);
      if (p.maxLag) setMaxLag(p.maxLag);
      if (p.labelStatus) setLabelStatus(p.labelStatus);
      if (p.flagPct !== 5) setFlagPct(p.flagPct);
      if (p.tab) setActiveTab(p.tab);
    }
  }, []);

  // Debounce MP name search
  useEffect(() => {
    if (!isMounted) return;
    const timer = setTimeout(() => {
      if (searchInput !== debouncedSearch) {
        setDebouncedSearch(searchInput);
        setPage(1);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput, debouncedSearch, isMounted]);

  // Sync to URL search params
  useEffect(() => {
    if (!isMounted || typeof window === "undefined") return;
    const sp = new URLSearchParams();
    if (method !== "gat") sp.set("method", method);
    if (page > 1) sp.set("page", String(page));
    if (debouncedSearch.trim()) sp.set("search", debouncedSearch.trim());
    if (selectedState.trim()) sp.set("state", selectedState.trim());
    if (selectedDistrict.trim()) sp.set("district", selectedDistrict.trim());
    if (selectedStatus.trim()) sp.set("status", selectedStatus.trim());
    if (minAmount.trim()) sp.set("min_amount", minAmount.trim());
    if (maxAmount.trim()) sp.set("max_amount", maxAmount.trim());
    if (minLag.trim()) sp.set("min_lag", minLag.trim());
    if (maxLag.trim()) sp.set("max_lag", maxLag.trim());
    if (labelStatus.trim()) sp.set("label_status", labelStatus.trim());
    if (flagPct !== 5) sp.set("flag_pct", String(flagPct));
    if (activeTab !== "overview") sp.set("tab", activeTab);

    const qs = sp.toString();
    const targetUrl = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
    window.history.replaceState(null, "", targetUrl);
  }, [
    isMounted,
    method,
    page,
    debouncedSearch,
    selectedState,
    selectedDistrict,
    selectedStatus,
    minAmount,
    maxAmount,
    minLag,
    maxLag,
    labelStatus,
    flagPct,
    activeTab,
  ]);

  // Handle browser back/forward buttons
  useEffect(() => {
    const handlePopState = () => {
      const p = parseUrlParams();
      if (!p) return;
      setMethod(p.method);
      setPage(p.page);
      setSearchInput(p.search);
      setDebouncedSearch(p.search);
      setSelectedState(p.state);
      setSelectedDistrict(p.district);
      setSelectedStatus(p.status);
      setMinAmount(p.minAmount);
      setMaxAmount(p.maxAmount);
      setMinLag(p.minLag);
      setMaxLag(p.maxLag);
      setLabelStatus(p.labelStatus);
      setFlagPct(p.flagPct);
      if (p.tab) setActiveTab(p.tab);
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const overviewQuery = useQuery({
    queryKey: ["overview", method, flagPct],
    queryFn: () => fetchOverview(method, flagPct),
  });

  const geographyQuery = useQuery({
    queryKey: ["geography", method, flagPct],
    queryFn: () => fetchGeography(method, flagPct),
  });

  const filterOptionsQuery = useQuery({
    queryKey: ["filter-options"],
    queryFn: fetchFilterOptions,
  });
  const filterOptions = filterOptionsQuery.data?.data;

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

  const activeFilters: RiskScoresFilterParams = useMemo(
    () => ({
      search: debouncedSearch,
      state: selectedState,
      district: selectedDistrict,
      status: selectedStatus,
      min_amount: minAmount,
      max_amount: maxAmount,
      min_lag: minLag,
      max_lag: maxLag,
      label_status: labelStatus,
      flag_pct: flagPct,
    }),
    [
      debouncedSearch,
      selectedState,
      selectedDistrict,
      selectedStatus,
      minAmount,
      maxAmount,
      minLag,
      maxLag,
      labelStatus,
      flagPct,
    ],
  );

  const scores = useQuery({
    queryKey: [
      "risk-scores",
      method,
      page,
      debouncedSearch,
      selectedState,
      selectedDistrict,
      selectedStatus,
      minAmount,
      maxAmount,
      minLag,
      maxLag,
      labelStatus,
      flagPct,
    ],
    queryFn: () => fetchRiskScores(method, page, PAGE_SIZE, activeFilters),
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

  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth });
  const labellingEnabled = health.data?.data.labelling_enabled ?? false;

  const isAnyMock =
    scores.data?.source === "mock" ||
    methodsQuery.data?.source === "mock" ||
    graph.data?.source === "mock" ||
    stats.data?.source === "mock" ||
    explain.data?.source === "mock" ||
    labelTotals.data?.source === "mock" ||
    overviewQuery.data?.source === "mock" ||
    geographyQuery.data?.source === "mock" ||
    health.data?.source === "mock";

  const label = useMutation({
    mutationFn: ({ workIndex, value }: { workIndex: number; value: LabelValue }) => {
      if (isAnyMock) {
        return Promise.reject(new Error("Labelling disabled in demo mode."));
      }
      if (!labellingEnabled) {
        return Promise.reject(new Error("Labelling is disabled in this deployment."));
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
              ? "Flagged for review"
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
  const cutoffScore =
    scores.data?.data.cutoff_score ??
    METHOD_THRESHOLDS[method as ScoringMethod]?.[flagPct as 1 | 5 | 10];
  const thresholdRank = scores.data?.data.threshold_rank ?? Math.round(15000 * (flagPct / 100));
  const flaggedCount = thresholdRank;

  const onLabel = (workIndex: number, value: LabelValue) => {
    if (isAnyMock || !labellingEnabled) return;
    label.mutate({ workIndex, value });
  };

  const hasActiveFilters = Boolean(
    searchInput.trim() ||
      selectedState ||
      selectedDistrict ||
      selectedStatus ||
      minAmount ||
      maxAmount ||
      minLag ||
      maxLag ||
      labelStatus,
  );

  const clearFilters = () => {
    setSearchInput("");
    setDebouncedSearch("");
    setSelectedState("");
    setSelectedDistrict("");
    setSelectedStatus("");
    setMinAmount("");
    setMaxAmount("");
    setMinLag("");
    setMaxLag("");
    setLabelStatus("");
    setPage(1);
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
                  ["Flagged for review", totals.s, "text-alert"],
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

        {/* Top-Level Navigation Tabs */}
        <nav className="border-b border-border bg-card">
          <div className="mx-auto flex max-w-[1600px] gap-1 px-6">
            <button
              onClick={() => setActiveTab("overview")}
              className={cn(
                "border-b-2 px-5 py-3 text-xs font-semibold uppercase tracking-wider transition-colors cursor-pointer",
                activeTab === "overview"
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-border",
              )}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab("geography")}
              className={cn(
                "border-b-2 px-5 py-3 text-xs font-semibold uppercase tracking-wider transition-colors cursor-pointer",
                activeTab === "geography"
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-border",
              )}
            >
              Geography
            </button>
            <button
              onClick={() => setActiveTab("triage")}
              className={cn(
                "border-b-2 px-5 py-3 text-xs font-semibold uppercase tracking-wider transition-colors cursor-pointer",
                activeTab === "triage"
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-border",
              )}
            >
              Triage Console
            </button>
            <button
              onClick={() => setActiveTab("evaluation")}
              className={cn(
                "border-b-2 px-5 py-3 text-xs font-semibold uppercase tracking-wider transition-colors cursor-pointer",
                activeTab === "evaluation"
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-border",
              )}
            >
              Evaluation & Benchmarks
            </button>
          </div>
        </nav>

        {activeTab === "overview" ? (
          <main className="mx-auto max-w-[1600px] px-6 py-6">
            <OverviewTab
              data={overviewQuery.data?.data ?? null}
              isLoading={overviewQuery.isLoading}
              isMock={overviewQuery.data?.source === "mock"}
              error={overviewQuery.data?.error}
              method={method}
              setMethod={setMethod}
              flagPct={flagPct}
              setFlagPct={setFlagPct}
              methodList={methodList}
            />
          </main>
        ) : activeTab === "geography" ? (
          <main className="mx-auto max-w-[1600px] px-6 py-6">
            <GeographyTab
              data={geographyQuery.data?.data ?? null}
              isLoading={geographyQuery.isLoading}
              isMock={geographyQuery.data?.source === "mock"}
              error={geographyQuery.data?.error}
              method={method}
              setMethod={setMethod}
              flagPct={flagPct}
              setFlagPct={setFlagPct}
              methodList={methodList}
              onSelectAgency={(ida) => setSelectedAgencyId(ida)}
            />
          </main>
        ) : activeTab === "evaluation" ? (
          <main className="mx-auto max-w-[1600px] px-6 py-6">
            <EvaluationTab />
          </main>
        ) : (
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
                    Showing {total.toLocaleString("en-IN")} of 15,000 · page {page} of{" "}
                    {pageCount.toLocaleString("en-IN")}
                  </p>
                  {!labellingEnabled && (
                    <p className="mt-1 text-xs text-muted-foreground">
                      Labels shown are the 60 analyst labels collected offline; labelling is disabled in this deployment.
                    </p>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-4">
                  {/* Scoring Method */}
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

                  {/* Review Cutoff Selector */}
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] tracking-wider text-muted-foreground uppercase">
                      Review Cutoff
                    </span>
                    <div className="flex border border-border">
                      {([1, 5, 10] as const).map((pct) => (
                        <button
                          key={pct}
                          onClick={() => {
                            setFlagPct(pct);
                            setPage(1);
                          }}
                          className={cn(
                            "px-2.5 py-1.5 text-xs font-medium cursor-pointer transition-colors",
                            flagPct === pct
                              ? "bg-navy text-navy-foreground"
                              : "text-muted-foreground hover:bg-muted",
                          )}
                        >
                          Top {pct}%
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Dynamic Review Cutoff Threshold Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-secondary/30 px-5 py-2.5 text-xs">
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Review Cutoff:
                    </span>
                    <span className="font-semibold text-foreground">Top {flagPct}%</span>
                    <span className="text-muted-foreground font-mono">
                      ({thresholdRank.toLocaleString("en-IN")} works)
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Score Threshold:
                    </span>
                    <span className="font-mono font-semibold text-foreground">
                      {currentMethod?.direction === "lower_is_riskier" ? "≤ " : "≥ "}
                      {cutoffScore !== undefined ? cutoffScore.toFixed(3) : "—"}
                      {currentMethod?.direction === "lower_is_riskier" ? " distance" : ""}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Cutoff Band:
                    </span>
                    <span className="font-medium text-foreground">Top {flagPct}%</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      if (labelStatus === "FLAGGED") {
                        setLabelStatus("");
                      } else {
                        setLabelStatus("FLAGGED");
                      }
                      setPage(1);
                    }}
                    className={cn(
                      "cursor-pointer border px-2.5 py-1 text-xs font-medium transition-colors",
                      labelStatus === "FLAGGED"
                        ? "border-primary bg-primary text-primary-foreground font-semibold shadow-xs"
                        : "border-border bg-card text-muted-foreground hover:bg-muted hover:text-foreground",
                    )}
                  >
                    {labelStatus === "FLAGGED"
                      ? `Viewing Top ${flagPct}% Flagged Only (${total.toLocaleString("en-IN")})`
                      : `Filter to Top ${flagPct}% Cutoff Only (${thresholdRank.toLocaleString("en-IN")})`}
                  </button>
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

              {/* Disclaimer Banner */}
              <div className="border-b border-border bg-secondary/30 px-5 py-2 text-xs text-muted-foreground">
                Scores are for audit prioritisation only and are based on model output.
              </div>

              {/* Filter Controls Bar */}
              <div className="border-b border-border bg-card px-5 py-3 space-y-3">
                <div className="flex flex-wrap items-end gap-3">
                  {/* Search */}
                  <div className="flex-1 min-w-[200px]">
                    <label
                      htmlFor="filter-search"
                      className="block text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-1"
                    >
                      MP Search
                    </label>
                    <input
                      id="filter-search"
                      type="text"
                      value={searchInput}
                      onChange={(e) => setSearchInput(e.target.value)}
                      placeholder="Search MP name..."
                      className="w-full border border-border bg-background px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-ring"
                    />
                  </div>

                  {/* State */}
                  <div className="w-[170px]">
                    <label
                      htmlFor="filter-state"
                      className="block text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-1"
                    >
                      State
                    </label>
                    <select
                      id="filter-state"
                      value={selectedState}
                      onChange={(e) => {
                        const val = e.target.value;
                        setSelectedState(val);
                        setSelectedDistrict("");
                        setPage(1);
                      }}
                      className="w-full border border-border bg-background px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                    >
                      <option value="">All States</option>
                      {(filterOptions?.states ?? []).map((s) => (
                        <option key={s} value={s}>
                          {s}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* District (depends on State) */}
                  <div className="w-[170px]">
                    <label
                      htmlFor="filter-district"
                      className="block text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-1"
                    >
                      District
                    </label>
                    <select
                      id="filter-district"
                      value={selectedDistrict}
                      disabled={!selectedState}
                      onChange={(e) => {
                        setSelectedDistrict(e.target.value);
                        setPage(1);
                      }}
                      className="w-full border border-border bg-background px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <option value="">
                        {selectedState ? "All Districts" : "Select a State first"}
                      </option>
                      {selectedState &&
                        (filterOptions?.districts_by_state?.[selectedState] ?? []).map((d) => (
                          <option key={d} value={d}>
                            {d}
                          </option>
                        ))}
                    </select>
                  </div>

                  {/* Status */}
                  <div className="w-[160px]">
                    <label
                      htmlFor="filter-status"
                      className="block text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-1"
                    >
                      Status
                    </label>
                    <select
                      id="filter-status"
                      value={selectedStatus}
                      onChange={(e) => {
                        setSelectedStatus(e.target.value);
                        setPage(1);
                      }}
                      className="w-full border border-border bg-background px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                    >
                      <option value="">All Statuses</option>
                      {(filterOptions?.statuses ?? []).map((st) => (
                        <option key={st} value={st}>
                          {st}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Label Status */}
                  <div className="w-[140px]">
                    <label
                      htmlFor="filter-label-status"
                      className="block text-[10px] font-medium uppercase tracking-wider text-muted-foreground mb-1"
                    >
                      Label Status
                    </label>
                    <select
                      id="filter-label-status"
                      value={labelStatus}
                      onChange={(e) => {
                        setLabelStatus(e.target.value);
                        setPage(1);
                      }}
                      className="w-full border border-border bg-background px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                    >
                      <option value="">All Labels</option>
                      <option value="FLAGGED">Flagged for Review (Top {flagPct}%)</option>
                      <option value="TRAINING LABEL">Training Label</option>
                      <option value="GAT-TOP">GAT-Top</option>
                      <option value="NEW">New</option>
                    </select>
                  </div>
                </div>

                {/* Second Row: Amount and Lag ranges + Clear Filters + Showing X of 15,000 */}
                <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-border/50 text-xs">
                  <div className="flex flex-wrap items-center gap-4">
                    {/* Amount Range */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                        Amount (₹):
                      </span>
                      <input
                        type="number"
                        placeholder="Min ₹"
                        value={minAmount}
                        onChange={(e) => {
                          setMinAmount(e.target.value);
                          setPage(1);
                        }}
                        className="w-24 border border-border bg-background px-2 py-1 text-xs text-foreground font-mono placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-ring"
                      />
                      <span className="text-muted-foreground">–</span>
                      <input
                        type="number"
                        placeholder="Max ₹"
                        value={maxAmount}
                        onChange={(e) => {
                          setMaxAmount(e.target.value);
                          setPage(1);
                        }}
                        className="w-24 border border-border bg-background px-2 py-1 text-xs text-foreground font-mono placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-ring"
                      />
                    </div>

                    {/* Lag Range */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                        Lag (days):
                      </span>
                      <input
                        type="number"
                        placeholder="Min"
                        value={minLag}
                        onChange={(e) => {
                          setMinLag(e.target.value);
                          setPage(1);
                        }}
                        className="w-16 border border-border bg-background px-2 py-1 text-xs text-foreground font-mono placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-ring"
                      />
                      <span className="text-muted-foreground">–</span>
                      <input
                        type="number"
                        placeholder="Max"
                        value={maxLag}
                        onChange={(e) => {
                          setMaxLag(e.target.value);
                          setPage(1);
                        }}
                        className="w-16 border border-border bg-background px-2 py-1 text-xs text-foreground font-mono placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-ring"
                      />
                    </div>

                    {/* Clear filters button */}
                    <button
                      onClick={clearFilters}
                      disabled={!hasActiveFilters}
                      className={cn(
                        "border border-border px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer",
                        hasActiveFilters
                          ? "text-foreground hover:bg-muted"
                          : "opacity-40 cursor-not-allowed text-muted-foreground",
                      )}
                    >
                      Clear filters
                    </button>
                  </div>

                  {/* Showing X of 15,000 */}
                  <div className="flex items-center gap-2">
                    <span className="border border-border bg-secondary/60 px-2.5 py-1 text-xs font-medium text-foreground tabular-nums">
                      Showing {total.toLocaleString("en-IN")} of 15,000
                    </span>
                  </div>
                </div>
              </div>

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
                  labellingEnabled={labellingEnabled}
                  method={method}
                  direction={currentMethod?.direction}
                  flagPct={flagPct}
                  cutoffScore={cutoffScore}
                  thresholdRank={thresholdRank}
                  rowFeedback={rowFeedback}
                  onSelect={setSelected}
                  onLabel={onLabel}
                  onSelectAgency={(ida) => setSelectedAgencyId(ida)}
                />
              )}

              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-5 py-3 text-xs">
                <span className="text-muted-foreground">
                  {total === 0
                    ? "No matching records"
                    : `Rows ${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, total)}`}
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
                labellingEnabled={labellingEnabled}
                flagPct={flagPct}
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
        )}

        <AgencyDrawer
          ida={selectedAgencyId}
          onClose={() => setSelectedAgencyId(null)}
          method={method}
          flagPct={flagPct}
        />
      </div>

      <footer className="border-t border-border bg-card py-5 text-center text-xs text-muted-foreground">
        Flags are leads for review, not proof of misconduct.
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
