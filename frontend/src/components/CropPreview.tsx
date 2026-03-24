import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import { useI18n } from "../i18n";

interface Props {
  projectId: string;
}

export function CropPreview({ projectId }: Props) {
  const { t } = useI18n();
  const [crops, setCrops] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<number | null>(null);

  const loadCrops = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listCrops(projectId);
      setCrops(res.crops);
    } catch {
      setCrops([]);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadCrops();
  }, [loadCrops]);

  if (loading) {
    return <p style={{ color: "#888" }}>{t.cropLoading}</p>;
  }

  if (crops.length === 0) {
    return <p style={{ color: "#888" }}>{t.cropEmpty}</p>;
  }

  return (
    <div>
      {/* Lightbox overlay */}
      {selected !== null && (
        <div
          onClick={() => setSelected(null)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.85)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            cursor: "zoom-out",
          }}
        >
          <img
            src={api.getCropUrl(projectId, crops[selected])}
            alt={crops[selected]}
            style={{
              maxWidth: "90vw",
              maxHeight: "90vh",
              borderRadius: 8,
              objectFit: "contain",
            }}
          />
          <div
            style={{
              position: "absolute",
              bottom: 24,
              color: "#ccc",
              fontSize: 14,
            }}
          >
            {crops[selected]} ({selected + 1}/{crops.length})
          </div>
          {/* Navigate prev/next */}
          {selected > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelected(selected - 1);
              }}
              style={navBtn("left")}
            >
              &lt;
            </button>
          )}
          {selected < crops.length - 1 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelected(selected + 1);
              }}
              style={navBtn("right")}
            >
              &gt;
            </button>
          )}
        </div>
      )}

      {/* Thumbnail grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 8,
        }}
      >
        {crops.map((name, i) => (
          <div
            key={name}
            onClick={() => setSelected(i)}
            style={{
              cursor: "zoom-in",
              borderRadius: 6,
              overflow: "hidden",
              border: "1px solid #333",
              position: "relative",
            }}
          >
            <img
              src={api.getCropUrl(projectId, name)}
              alt={name}
              loading="lazy"
              style={{ width: "100%", display: "block" }}
            />
            <div
              style={{
                position: "absolute",
                bottom: 0,
                left: 0,
                right: 0,
                padding: "4px 6px",
                background: "rgba(0,0,0,0.6)",
                fontSize: 11,
                color: "#ccc",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {name}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function navBtn(side: "left" | "right"): React.CSSProperties {
  return {
    position: "absolute",
    [side]: 16,
    top: "50%",
    transform: "translateY(-50%)",
    background: "rgba(255,255,255,0.15)",
    border: "none",
    color: "#fff",
    fontSize: 28,
    padding: "8px 14px",
    borderRadius: 8,
    cursor: "pointer",
  };
}
