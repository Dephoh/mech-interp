import { useMemo, useState } from "react";
import { useTestLabStore } from "../../store/testlabStore";

const CATEGORIES = ["All", "keyword", "ir_type", "card_type", "individual"] as const;

interface Props {
  onClose: () => void;
  onSelect: (index: number) => void;
}

export function ScenarioBrowser({ onClose, onSelect }: Props) {
  const scenarios = useTestLabStore((s) => s.scenarios);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<string>("All");

  const filtered = useMemo(() => {
    const lower = search.toLowerCase();
    return scenarios
      .map((s, i) => ({ ...s, _index: i }))
      .filter((s) => {
        if (category !== "All" && s.category !== category) return false;
        if (!lower) return true;
        return (
          s.name.toLowerCase().includes(lower) ||
          s.description.toLowerCase().includes(lower) ||
          s.tags.some((t) => t.toLowerCase().includes(lower))
        );
      });
  }, [scenarios, search, category]);

  return (
    <div className="scenario-browser-overlay" onClick={onClose}>
      <div className="scenario-browser" onClick={(e) => e.stopPropagation()}>
        <div className="scenario-browser__header">
          <input
            className="scenario-browser__search"
            placeholder="Search scenarios..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            autoFocus
          />
          <button className="scenario-browser__close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="scenario-browser__categories">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              className={`scenario-browser__cat-btn ${category === cat ? "scenario-browser__cat-btn--active" : ""}`}
              onClick={() => setCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="scenario-browser__list">
          {filtered.length === 0 && (
            <div style={{ padding: 16, color: "var(--text-dim)", textAlign: "center" }}>
              No matching scenarios.
            </div>
          )}
          {filtered.map((s) => (
            <div
              key={s.scenario_id}
              className="scenario-browser__item"
              onClick={() => onSelect(s._index)}
            >
              <div className="scenario-browser__item-name">{s.name}</div>
              <div className="scenario-browser__item-desc">{s.description}</div>
              {s.tags.length > 0 && (
                <div className="scenario-browser__item-tags">
                  {s.tags.map((t) => (
                    <span key={t} className="scenario-browser__tag">{t}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
