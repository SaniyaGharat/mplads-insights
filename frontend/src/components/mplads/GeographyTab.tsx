import { useState, useMemo } from "react";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Building2,
  ChevronRight,
  Info,
} from "lucide-react";
import {
  formatINR,
  type GeographyResponse,
  type StateGeoItem,
  type DistrictGeoItem,
  type ScoringMethod,
  type MethodInfo,
  DEFAULT_METHODS,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { MapChart } from "./MapChart";

interface Props {
  data: GeographyResponse | null;
  isLoading: boolean;
  isMock: boolean;
  error?: string | undefined;
  method: ScoringMethod;
  setMethod: (m: ScoringMethod) => void;
  flagPct: 1 | 5 | 10;
  setFlagPct: (pct: 1 | 5 | 10) => void;
  methodList?: MethodInfo[] | undefined;
  onSelectAgency: (ida: string) => void;
}

type StateSortField = "state" | "works" | "total_amount" | "flagged_count" | "flagged_share";
type DistrictSortField = "district" | "works" | "total_amount" | "flagged_count" | "flagged_share";

export function GeographyTab({
  data,
  isLoading,
  isMock,
  error,
  method,
  setMethod,
  flagPct,
  setFlagPct,
  methodList = DEFAULT_METHODS,
  onSelectAgency,
}: Props) {
  // Default sort is by works descending, NOT by flagged share
  const [stateSortCol, setStateSortCol] = useState<StateSortField>("works");
  const [stateSortAsc, setStateSortAsc] = useState<boolean>(false);

  const [selectedStateName, setSelectedStateName] = useState<string | null>(null);

  const [distSortCol, setDistSortCol] = useState<DistrictSortField>("works");
  const [distSortAsc, setDistSortAsc] = useState<boolean>(false);

  const currentMethodInfo =
    methodList?.find((m) => m.id === method) ?? DEFAULT_METHODS.find((m) => m.id === method);

  const handleStateSort = (col: StateSortField) => {
    if (stateSortCol === col) {
      setStateSortAsc(!stateSortAsc);
    } else {
      setStateSortCol(col);
      setStateSortAsc(col === "state"); // alphabetical ascending by default, numeric descending
    }
  };

  const handleDistSort = (col: DistrictSortField) => {
    if (distSortCol === col) {
      setDistSortAsc(!distSortAsc);
    } else {
      setDistSortCol(col);
      setDistSortAsc(col === "district");
    }
  };

  // Sort states
  const sortedStates = useMemo(() => {
    if (!data?.states) return [];
    return [...data.states].sort((a, b) => {
      let cmp = 0;
      if (stateSortCol === "state") {
        cmp = a.state.localeCompare(b.state);
      } else {
        cmp = a[stateSortCol] - b[stateSortCol];
      }
      return stateSortAsc ? cmp : -cmp;
    });
  }, [data?.states, stateSortCol, stateSortAsc]);

  // Selected state
  const selectedState: StateGeoItem | undefined = useMemo(() => {
    if (!sortedStates.length) return undefined;
    if (!selectedStateName) return sortedStates[0];
    return sortedStates.find((s) => s.state === selectedStateName) ?? sortedStates[0];
  }, [sortedStates, selectedStateName]);

  // Sort districts of selected state
  const sortedDistricts = useMemo(() => {
    if (!selectedState?.districts) return [];
    return [...selectedState.districts].sort((a, b) => {
      let cmp = 0;
      if (distSortCol === "district") {
        cmp = a.district.localeCompare(b.district);
      } else {
        cmp = a[distSortCol] - b[distSortCol];
      }
      return distSortAsc ? cmp : -cmp;
    });
  }, [selectedState?.districts, distSortCol, distSortAsc]);

  const renderSortIcon = (active: boolean, asc: boolean) => {
    if (!active) return <ArrowUpDown className="ml-1 inline h-3 w-3 opacity-40" />;
    return asc ? <ArrowUp className="ml-1 inline h-3 w-3 text-primary" /> : <ArrowDown className="ml-1 inline h-3 w-3 text-primary" />;
  };

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
          <span>State and district level audit allocation view</span>
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
            {(methodList ?? DEFAULT_METHODS).map((m) => {
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

      {/* Analytical Guidance Note */}
      <div className="border border-border/70 bg-secondary/40 px-5 py-3 text-xs text-muted-foreground">
        <p className="flex items-center gap-2">
          <Info className="h-4 w-4 text-primary shrink-0" />
          <span>
            <strong className="text-foreground">Statistical context:</strong> Larger states have more flagged works simply because they have more works.
          </span>
        </p>
      </div>

      {/* Method Note */}
      {currentMethodInfo && (
        <p className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground">{currentMethodInfo.name}:</span>{" "}
          {currentMethodInfo.note}
        </p>
      )}

      {/* India State Bubble Map */}
      <MapChart
        states={data?.states ?? []}
        selectedState={selectedStateName ?? sortedStates[0]?.state ?? null}
        onSelectState={(st) => setSelectedStateName(st)}
        flagPct={flagPct}
        isMock={isMock}
      />

      {/* Main Dual Table Layout: States Table and Selected State Districts Table */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* States Table (Left / 7 cols) */}
        <div className="lg:col-span-7 space-y-3">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold tracking-tight">
              State Summary Table ({sortedStates.length})
            </h3>
            <span className="text-xs text-muted-foreground">
              Select a state row to inspect its districts
            </span>
          </div>

          <div className="overflow-x-auto border border-border bg-card shadow-xs">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border bg-secondary/50 font-medium text-muted-foreground">
                <tr>
                  <th
                    className="cursor-pointer px-4 py-3 hover:text-foreground"
                    onClick={() => handleStateSort("state")}
                  >
                    State {renderSortIcon(stateSortCol === "state", stateSortAsc)}
                  </th>
                  <th
                    className="cursor-pointer px-4 py-3 text-right hover:text-foreground"
                    onClick={() => handleStateSort("works")}
                  >
                    Works (n) {renderSortIcon(stateSortCol === "works", stateSortAsc)}
                  </th>
                  <th
                    className="cursor-pointer px-4 py-3 text-right hover:text-foreground"
                    onClick={() => handleStateSort("total_amount")}
                  >
                    Total Amount {renderSortIcon(stateSortCol === "total_amount", stateSortAsc)}
                  </th>
                  <th
                    className="cursor-pointer px-4 py-3 text-right hover:text-foreground"
                    onClick={() => handleStateSort("flagged_count")}
                  >
                    Flagged for Review {renderSortIcon(stateSortCol === "flagged_count", stateSortAsc)}
                  </th>
                  <th
                    className="cursor-pointer px-4 py-3 text-right hover:text-foreground"
                    onClick={() => handleStateSort("flagged_share")}
                  >
                    Flagged Share (%) {renderSortIcon(stateSortCol === "flagged_share", stateSortAsc)}
                  </th>
                  <th className="w-8 px-2 py-3"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {isLoading ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-12 text-center text-muted-foreground">
                      Loading states data…
                    </td>
                  </tr>
                ) : sortedStates.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-12 text-center text-muted-foreground">
                      No state records available.
                    </td>
                  </tr>
                ) : (
                  sortedStates.map((st) => {
                    const isSelected = selectedState?.state === st.state;
                    const isSmall = st.works < 20;
                    return (
                      <tr
                        key={st.state}
                        onClick={() => setSelectedStateName(st.state)}
                        className={cn(
                          "cursor-pointer transition-colors hover:bg-secondary/40",
                          isSelected ? "bg-primary/10 font-medium" : "",
                        )}
                      >
                        <td className="px-4 py-2.5 font-medium text-foreground">
                          {st.state}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-foreground">
                          {st.works.toLocaleString("en-IN")}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-muted-foreground">
                          {isMock ? "—" : formatINR(st.total_amount)}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-alert font-semibold">
                          {st.flagged_count.toLocaleString("en-IN")}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono">
                          {isSmall ? (
                            <span
                              title="small group: share is unreliable"
                              className="text-muted-foreground/60 cursor-help underline decoration-dotted"
                            >
                              {st.flagged_share}% (n={st.works})
                            </span>
                          ) : (
                            <span className="text-foreground">
                              {st.flagged_share}% <span className="text-muted-foreground text-[10px]">(n={st.works})</span>
                            </span>
                          )}
                        </td>
                        <td className="px-2 py-2.5 text-center text-muted-foreground">
                          {isSelected && <ChevronRight className="h-4 w-4 text-primary" />}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* District Table (Right / 5 cols) */}
        <div className="lg:col-span-5 space-y-3">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold tracking-tight">
              Districts in {selectedState?.state ?? "Selected State"} ({sortedDistricts.length})
            </h3>
            <span className="text-xs text-muted-foreground">
              Built from agency district prefix
            </span>
          </div>

          <div className="overflow-x-auto border border-border bg-card shadow-xs">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border bg-secondary/50 font-medium text-muted-foreground">
                <tr>
                  <th
                    className="cursor-pointer px-3 py-3 hover:text-foreground"
                    onClick={() => handleDistSort("district")}
                  >
                    District {renderSortIcon(distSortCol === "district", distSortAsc)}
                  </th>
                  <th
                    className="cursor-pointer px-3 py-3 text-right hover:text-foreground"
                    onClick={() => handleDistSort("works")}
                  >
                    Works (n) {renderSortIcon(distSortCol === "works", distSortAsc)}
                  </th>
                  <th
                    className="cursor-pointer px-3 py-3 text-right hover:text-foreground"
                    onClick={() => handleDistSort("flagged_count")}
                  >
                    Flagged {renderSortIcon(distSortCol === "flagged_count", distSortAsc)}
                  </th>
                  <th
                    className="cursor-pointer px-3 py-3 text-right hover:text-foreground"
                    onClick={() => handleDistSort("flagged_share")}
                  >
                    Share (%) {renderSortIcon(distSortCol === "flagged_share", distSortAsc)}
                  </th>
                  <th className="px-3 py-3 text-center">Agency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {isLoading ? (
                  <tr>
                    <td colSpan={5} className="px-3 py-12 text-center text-muted-foreground">
                      Loading districts…
                    </td>
                  </tr>
                ) : sortedDistricts.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-3 py-12 text-center text-muted-foreground">
                      No district records for this state.
                    </td>
                  </tr>
                ) : (
                  sortedDistricts.map((d) => {
                    const isSmall = d.works < 20;
                    return (
                      <tr
                        key={d.district}
                        className="transition-colors hover:bg-secondary/30"
                      >
                        <td className="px-3 py-2.5 font-medium text-foreground">
                          {d.district}
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono text-foreground">
                          {d.works}
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono text-alert font-semibold">
                          {d.flagged_count}
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono">
                          {isSmall ? (
                            <span
                              title="small group: share is unreliable"
                              className="text-muted-foreground/60 cursor-help underline decoration-dotted"
                            >
                              {d.flagged_share}% (n={d.works})
                            </span>
                          ) : (
                            <span className="text-foreground">
                              {d.flagged_share}% <span className="text-muted-foreground text-[10px]">(n={d.works})</span>
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2.5 text-center">
                          <button
                            onClick={() => onSelectAgency(d.primary_ida || d.district)}
                            className="inline-flex items-center gap-1 border border-border bg-card px-2 py-1 text-[11px] font-medium text-primary hover:bg-secondary hover:text-foreground cursor-pointer transition-colors"
                            title="Open Agency Profile Drawer"
                          >
                            <Building2 className="h-3 w-3" />
                            Agency
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
