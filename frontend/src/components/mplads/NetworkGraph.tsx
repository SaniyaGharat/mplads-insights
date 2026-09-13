import { useMemo } from "react";
import type { GraphResponse } from "@/lib/api";

interface Props {
  graph: GraphResponse | null;
  isLoading: boolean;
  isMock: boolean;
  selectedWorkIndex: number | null;
  onSelectEdge: (workIndex: number) => void;
}

const W = 900;
const H = 460;

export function NetworkGraph({
  graph,
  isLoading,
  isMock,
  selectedWorkIndex,
  onSelectEdge,
}: Props) {
  const layout = useMemo(() => {
    if (!graph) return null;
    const mps = graph.nodes.filter((n) => n.type === "MP");
    const idas = graph.nodes.filter((n) => n.type === "IDA");
    const pos = new Map<string, { x: number; y: number }>();
    const place = (ids: typeof mps, x: number) => {
      const step = (H - 60) / Math.max(1, ids.length);
      ids.forEach((n, i) => pos.set(n.id, { x, y: 40 + step * (i + 0.5) }));
    };
    place(mps, 190);
    place(idas, W - 190);
    return { pos, mps, idas };
  }, [graph]);

  if (isLoading) {
    return (
      <div className="border border-border bg-card p-10 text-sm text-muted-foreground">
        Loading network…
      </div>
    );
  }
  if (!graph || !layout) {
    return (
      <div className="border border-border bg-card p-10 text-sm text-muted-foreground">
        Network data unavailable.
      </div>
    );
  }

  const maxScore = Math.max(...graph.edges.map((e) => e.score), 0.001);

  return (
    <div className="border border-border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3">
        <div>
          <h2 className="text-base font-semibold">Network of top-risk connections</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {layout.mps.length} MPs · {layout.idas.length} agencies · {graph.edges.length}{" "}
            sanctioned works
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <svg width="12" height="12">
              <circle cx="6" cy="6" r="5" className="fill-navy" />
            </svg>
            MP
          </span>
          <span className="flex items-center gap-1.5">
            <svg width="12" height="12">
              <rect x="1" y="1" width="10" height="10" className="fill-sage" />
            </svg>
            Implementing agency
          </span>
        </div>
      </div>

      <div className="overflow-x-auto p-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[700px] w-full">
          {graph.edges.map((e) => {
            const a = layout.pos.get(e.source);
            const b = layout.pos.get(e.target);
            if (!a || !b) return null;
            const active = selectedWorkIndex === e.work_index;
            return (
              <line
                key={e.work_index}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                className={active ? "stroke-alert" : "stroke-navy"}
                strokeOpacity={active ? 0.9 : 0.12 + (e.score / maxScore) * 0.28}
                strokeWidth={active ? 2 : 1}
                onClick={() => onSelectEdge(e.work_index)}
                style={{ cursor: "pointer" }}
              />
            );
          })}
          {layout.mps.map((n) => {
            const p = layout.pos.get(n.id)!;
            return (
              <g key={n.id}>
                <circle cx={p.x} cy={p.y} r={6} className="fill-navy" />
                <text
                  x={p.x - 12}
                  y={p.y + 4}
                  textAnchor="end"
                  className="fill-foreground text-[11px]"
                >
                  {n.id}
                </text>
              </g>
            );
          })}
          {layout.idas.map((n) => {
            const p = layout.pos.get(n.id)!;
            return (
              <g key={n.id}>
                <rect x={p.x - 5} y={p.y - 5} width={10} height={10} className="fill-sage" />
                <text x={p.x + 14} y={p.y + 4} className="fill-foreground text-[11px]">
                  {n.id}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      {isMock && (
        <p className="border-t border-border px-5 py-2 text-xs text-muted-foreground">
          Backend unreachable — network rendered from fallback sample.
        </p>
      )}
    </div>
  );
}
