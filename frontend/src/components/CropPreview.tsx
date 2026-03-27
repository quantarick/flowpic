import { useCallback, useEffect, useRef, useState } from "react";
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
  const [publishedFilenames, setPublishedFilenames] = useState<Set<string>>(new Set());
  const [feedback, setFeedback] = useState("");
  const [regenerating, setRegenerating] = useState(false);
  const [cacheBust, setCacheBust] = useState(0);
  const feedbackRef = useRef<HTMLInputElement>(null);

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
    api.getPublishedImages(projectId)
      .then((data) => {
        setPublishedFilenames(new Set(data.published_images.map((img) => img.crop_filename)));
      })
      .catch(() => {});
  }, [loadCrops, projectId]);

  const handleRegenerate = async () => {
    if (selected === null || !feedback.trim() || regenerating) return;
    setRegenerating(true);
    try {
      const res = await api.regenerateCrop(projectId, crops[selected], feedback.trim());
      setCrops((prev) => {
        const next = [...prev];
        next[selected] = res.crop_filename;
        return next;
      });
      setCacheBust((c) => c + 1);
      setFeedback("");
    } catch (err) {
      console.error("Regenerate failed:", err);
    } finally {
      setRegenerating(false);
    }
  };

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
          onClick={() => { setSelected(null); setFeedback(""); }}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.85)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            cursor: "zoom-out",
          }}
        >
          {/* Crop (main) + Original (small reference) */}
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              display: "flex",
              gap: 16,
              alignItems: "flex-end",
              justifyContent: "center",
              maxWidth: "92vw",
              maxHeight: "75vh",
              cursor: "default",
            }}
          >
            {/* Cropped image — main, large */}
            <img
              src={api.getCropUrl(projectId, crops[selected]) + `?v=${cacheBust}`}
              alt={crops[selected]}
              style={{
                maxWidth: "60vw",
                maxHeight: "70vh",
                borderRadius: 8,
                objectFit: "contain",
              }}
            />
            {/* Original image — small reference */}
            <div style={{ textAlign: "center", flexShrink: 0 }}>
              <div style={{ color: "#666", fontSize: 11, marginBottom: 4 }}>Original</div>
              <img
                src={api.getOriginalImageUrl(projectId, crops[selected])}
                alt="Original"
                style={{
                  maxWidth: "20vw",
                  maxHeight: "35vh",
                  borderRadius: 6,
                  objectFit: "contain",
                  border: "1px solid #444",
                  opacity: 0.85,
                }}
              />
            </div>
          </div>
          <div style={{ color: "#ccc", fontSize: 14, marginTop: 8 }}>
            {crops[selected]} ({selected + 1}/{crops.length})
          </div>
          {/* Feedback bar */}
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              display: "flex",
              gap: 8,
              marginTop: 12,
              width: "min(700px, 90vw)",
              cursor: "default",
            }}
          >
            <input
              ref={feedbackRef}
              type="text"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && feedback.trim() && !regenerating) {
                  handleRegenerate();
                }
              }}
              placeholder={t.cropFeedbackPlaceholder}
              disabled={regenerating}
              style={{
                flex: 1,
                padding: "8px 12px",
                borderRadius: 6,
                border: "1px solid #555",
                background: "#222",
                color: "#eee",
                fontSize: 14,
                outline: "none",
              }}
            />
            <button
              onClick={handleRegenerate}
              disabled={!feedback.trim() || regenerating}
              style={{
                padding: "8px 16px",
                borderRadius: 6,
                border: "none",
                background: regenerating ? "#555" : "#4a9eff",
                color: "#fff",
                fontSize: 14,
                cursor: regenerating || !feedback.trim() ? "not-allowed" : "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {regenerating ? t.regenerating : t.btnRegenerate}
            </button>
          </div>
          {/* Navigate prev/next */}
          {selected > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setSelected(selected - 1);
                setFeedback("");
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
                setFeedback("");
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
