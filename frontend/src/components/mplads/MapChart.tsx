import { useState, useMemo } from "react";
import { geoMercator, geoPath } from "d3-geo";
import { formatINR, type StateGeoItem } from "@/lib/api";
import { cn } from "@/lib/utils";
import indiaGeoRaw from "@/assets/india_states.json";

interface Props {
  states: StateGeoItem[];
  selectedState: string | null;
  onSelectState: (state: string) => void;
  flagPct: 1 | 5 | 10;
  isMock?: boolean;
}

// Canonical aliases to bridge works data names to GeoJSON st_nm properties
const STATE_ALIAS: Record<string, string> = {
  "Andaman & Nicobar Island": "Andaman and Nicobar Islands",
  "Andaman and Nicobar": "Andaman and Nicobar Islands",
  "Arunanchal Pradesh": "Arunachal Pradesh",
  "Dadara & Nagar Havelli": "Dadra and Nagar Haveli",
  "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
  "Daman & Diu": "Daman and Diu",
  "Jammu And Kashmir": "Jammu and Kashmir",
  "Jammu & Kashmir": "Jammu and Kashmir",
  "NCT of Delhi": "Delhi",
  "Delhi": "Delhi",
  "Orissa": "Odisha",
  "Odisha": "Odisha",
  "Uttaranchal": "Uttarakhand",
  "Uttarakhand": "Uttarakhand",
  "Telengana": "Telangana",
  "Telangana": "Telangana",
};

interface TooltipData {
  stateName: string;
  works: number;
  flaggedCount: number;
  flaggedShare: number;
  totalAmount: number;
  isSmall: boolean;
  x: number;
  y: number;
}

// Continuous color interpolator for Flagged Share %
// Palette: Low (0%) calm sage/slate -> Mid (flagPct%) warm amber -> High (2x flagPct%+) alert terracotta
function getShareColor(share: number, flagPct: number, isMock: boolean): string {
  if (isMock) return "rgb(156, 163, 175)"; // neutral grey for mock

  const norm = Math.max(0, Math.min(1, share / (flagPct * 2.2)));

  // 0.0: Sage Green rgb(74, 124, 89)
  // 0.5: Warm Amber  rgb(217, 149, 67)
  // 1.0: Terracotta  rgb(201, 74, 41)
  let r: number, g: number, b: number;
  if (norm <= 0.5) {
    const t = norm * 2;
    r = Math.round(74 + t * (217 - 74));
    g = Math.round(124 + t * (149 - 124));
    b = Math.round(89 + t * (67 - 89));
  } else {
    const t = (norm - 0.5) * 2;
    r = Math.round(217 + t * (201 - 217));
    g = Math.round(149 + t * (74 - 149));
    b = Math.round(67 + t * (41 - 67));
  }
  return `rgb(${r}, ${g}, ${b})`;
}

export function MapChart({
  states,
  selectedState,
  onSelectState,
  flagPct,
  isMock = false,
}: Props) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

  // SVG dimensions
  const svgWidth = 640;
  const svgHeight = 720;

  // d3-geo projection and path generator
  const { pathGenerator, featureList } = useMemo(() => {
    const proj = geoMercator().fitExtent(
      [
        [20, 20],
        [svgWidth - 20, svgHeight - 40],
      ],
      indiaGeoRaw as any,
    );
    const pathGen = geoPath().projection(proj);
    const feats = (indiaGeoRaw as any).features as any[];
    return { pathGenerator: pathGen, featureList: feats };
  }, []);

  // Map state name lookup
  const stateByGeoName = useMemo(() => {
    const map = new Map<string, StateGeoItem>();
    for (const item of states) {
      const geoName = STATE_ALIAS[item.state] ?? item.state;
      map.set(geoName.toLowerCase().trim(), item);
    }
    return map;
  }, [states]);

  // Compute matched and unmatched states
  const { matchedGeoFeatures, unmatchedWorksStates } = useMemo(() => {
    const matched = new Set<string>();
    const featuresWithData = featureList.map((f) => {
      const geoName = f.properties.st_nm as string;
      const dataItem = stateByGeoName.get(geoName.toLowerCase().trim());
      if (dataItem) matched.add(dataItem.state);
      const centroid = pathGenerator.centroid(f);
      return {
        feature: f,
        geoName,
        dataItem,
        centroid,
        pathD: pathGenerator(f) || "",
      };
    });

    const unmatched = states.filter((s) => !matched.has(s.state));
    return { matchedGeoFeatures: featuresWithData, unmatchedWorksStates: unmatched };
  }, [featureList, stateByGeoName, pathGenerator, states]);

  // Sqrt radius scale: max ~4,500 works -> 26px, min 3.5px
  const getRadius = (works: number) => {
    return Math.max(3.5, Math.sqrt(works / 4500) * 26);
  };

  return (
    <div className="relative flex flex-col border border-border bg-card p-4 shadow-xs">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-border pb-3">
        <div>
          <h3 className="text-sm font-semibold tracking-tight">Geographic Distribution of Sanctioned Works</h3>
          <p className="text-xs text-muted-foreground">
            Bubble size represents volume of works · Bubble color shows flagged share (%) at top {flagPct}% cutoff
          </p>
        </div>
        {isMock && (
          <span className="border border-alert/30 bg-alert/10 px-2 py-0.5 text-[11px] font-semibold text-alert">
            Sample Data (Demo Mode)
          </span>
        )}
      </div>

      <div className="relative mt-2 flex justify-center">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="h-auto w-full max-w-[620px] select-none"
          role="img"
          aria-label="Map of India showing MPLADS sanctioned works by state"
        >
          <defs>
            <linearGradient id="flaggedShareGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(74, 124, 89)" />
              <stop offset="50%" stopColor="rgb(217, 149, 67)" />
              <stop offset="100%" stopColor="rgb(201, 74, 41)" />
            </linearGradient>
          </defs>

          {/* 1. Base State Polygons */}
          <g className="state-boundaries">
            {matchedGeoFeatures.map(({ feature, pathD, dataItem }) => {
              const geoName = feature.properties.st_nm;
              const isSelected = selectedState && dataItem?.state === selectedState;
              return (
                <path
                  key={geoName}
                  d={pathD}
                  className={cn(
                    "transition-colors duration-150",
                    isSelected
                      ? "fill-primary/15 stroke-primary stroke-[1.5]"
                      : "fill-secondary/35 stroke-border hover:fill-secondary/70 stroke-[0.8]",
                  )}
                  onClick={() => {
                    if (dataItem) onSelectState(dataItem.state);
                  }}
                >
                  <title>{geoName}</title>
                </path>
              );
            })}
          </g>

          {/* 2. State Centroid Bubbles */}
          <g className="state-bubbles">
            {matchedGeoFeatures.map(({ geoName, dataItem, centroid }) => {
              if (!dataItem || isNaN(centroid[0]) || isNaN(centroid[1])) return null;

              const isSelected = selectedState === dataItem.state;
              const r = getRadius(dataItem.works);
              const fillColor = getShareColor(dataItem.flagged_share, flagPct, isMock);
              const isSmall = dataItem.works < 20;

              return (
                <g
                  key={geoName}
                  transform={`translate(${centroid[0]}, ${centroid[1]})`}
                  className="cursor-pointer transition-transform duration-100 hover:scale-110"
                  onClick={() => onSelectState(dataItem.state)}
                  onMouseEnter={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    setTooltip({
                      stateName: dataItem.state,
                      works: dataItem.works,
                      flaggedCount: dataItem.flagged_count,
                      flaggedShare: dataItem.flagged_share,
                      totalAmount: dataItem.total_amount,
                      isSmall,
                      x: centroid[0],
                      y: centroid[1] - r - 8,
                    });
                  }}
                  onMouseLeave={() => setTooltip(null)}
                >
                  {/* Selected ring indicator */}
                  {isSelected && (
                    <circle
                      r={r + 4}
                      fill="none"
                      stroke="var(--foreground)"
                      strokeWidth={2}
                      className="animate-pulse opacity-80"
                    />
                  )}

                  {/* Bubble circle */}
                  <circle
                    r={r}
                    fill={fillColor}
                    fillOpacity={isSmall ? 0.35 : 0.85}
                    stroke={isSelected ? "var(--foreground)" : isSmall ? "var(--muted-foreground)" : "#ffffff"}
                    strokeWidth={isSelected ? 2 : isSmall ? 1.2 : 0.8}
                    strokeDasharray={isSmall ? "3 2" : undefined}
                  />

                  {/* Centered label for large states */}
                  {dataItem.works >= 800 && (
                    <text
                      textAnchor="middle"
                      dy="0.3em"
                      className="pointer-events-none select-none font-mono text-[9px] font-bold fill-white drop-shadow-[0_1px_1px_rgba(0,0,0,0.8)]"
                    >
                      {dataItem.works >= 1000 ? `${(dataItem.works / 1000).toFixed(1)}k` : dataItem.works}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* 3. Embedded Map Legends */}
          <g transform="translate(24, 490)" className="map-legend">
            {/* Size Reference Legend */}
            <g className="size-legend">
              <text x={0} y={0} className="font-sans text-[11px] font-semibold fill-foreground">
                Works (n)
              </text>
              <g transform="translate(15, 20)">
                {[500, 1000, 2000].map((sampleWorks, idx) => {
                  const rad = getRadius(sampleWorks);
                  const cx = idx * 52 + 10;
                  return (
                    <g key={sampleWorks} transform={`translate(${cx}, 20)`}>
                      <circle
                        r={rad}
                        fill="none"
                        stroke="var(--foreground)"
                        strokeWidth={1}
                        className="opacity-60"
                      />
                      <text
                        y={rad + 14}
                        textAnchor="middle"
                        className="font-mono text-[10px] fill-muted-foreground"
                      >
                        {sampleWorks.toLocaleString("en-IN")}
                      </text>
                    </g>
                  );
                })}
              </g>
            </g>

            {/* Continuous Color Scale Legend */}
            <g transform="translate(0, 85)" className="color-legend">
              <text x={0} y={0} className="font-sans text-[11px] font-semibold fill-foreground">
                Flagged share (%)
              </text>
              <rect
                x={0}
                y={10}
                width={160}
                height={10}
                rx={2}
                fill={isMock ? "rgb(156, 163, 175)" : "url(#flaggedShareGrad)"}
                stroke="var(--border)"
              />
              <g transform="translate(0, 32)" className="font-mono text-[10px] fill-muted-foreground">
                <text x={0} textAnchor="start">
                  0%
                </text>
                <text x={80} textAnchor="middle">
                  {flagPct}% (Cutoff)
                </text>
                <text x={160} textAnchor="end">
                  {flagPct * 2}%+
                </text>
              </g>
            </g>

            {/* Small Group Legend */}
            <g transform="translate(0, 142)">
              <circle
                cx={7}
                cy={0}
                r={6}
                fill="rgb(156, 163, 175)"
                fillOpacity={0.35}
                stroke="var(--muted-foreground)"
                strokeDasharray="2 2"
                strokeWidth={1.2}
              />
              <text x={20} y={3} className="font-sans text-[10px] fill-muted-foreground">
                Faded dashed: &lt;20 works (small group)
              </text>
            </g>
          </g>

          {/* 4. Interactive Hover Tooltip */}
          {tooltip && (
            <g transform={`translate(${Math.min(svgWidth - 140, Math.max(130, tooltip.x))}, ${Math.max(80, tooltip.y)})`}>
              <rect
                x={-120}
                y={-64}
                width={240}
                height={78}
                rx={3}
                className="fill-card stroke-border filter drop-shadow-md"
                strokeWidth={1}
              />
              <text x={0} y={-46} textAnchor="middle" className="font-sans text-xs font-bold fill-foreground">
                {tooltip.stateName} {tooltip.isSmall ? "(Small Group)" : ""}
              </text>
              <text x={0} y={-30} textAnchor="middle" className="font-mono text-[11px] fill-muted-foreground">
                Works: <tspan className="font-semibold fill-foreground">{tooltip.works.toLocaleString("en-IN")}</tspan> · Flagged:{" "}
                <tspan className="font-semibold fill-alert">{tooltip.flaggedCount}</tspan> ({tooltip.flaggedShare}%)
              </text>
              <text x={0} y={-16} textAnchor="middle" className="font-mono text-[10px] fill-muted-foreground">
                Total Amount: {isMock ? "—" : formatINR(tooltip.totalAmount)}
              </text>
              <text x={0} y={-2} textAnchor="middle" className="font-sans text-[9px] font-medium fill-primary">
                Expected share at this cutoff: {flagPct}% (average across all works)
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Mandatory Statutory Note */}
      <div className="mt-3 border-t border-border pt-3 text-xs text-muted-foreground">
        <p className="leading-relaxed">
          Bubble size = number of works. Larger states have more flagged works simply because they have more works. Scores prioritise audit attention; they are not findings of misconduct.
        </p>

        {/* Visible Footnote for any unmatched states */}
        {unmatchedWorksStates.length > 0 && (
          <p className="mt-1 text-[11px] text-muted-foreground/80">
            <strong>Not shown:</strong> {unmatchedWorksStates.map((s) => s.state).join(", ")} (no boundary match).
          </p>
        )}
      </div>
    </div>
  );
}
