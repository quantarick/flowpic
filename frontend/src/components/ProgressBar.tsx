import type { ProgressMessage } from "../types";

interface Props {
  progress: ProgressMessage | null;
  onCancel?: () => void;
}

const STEP_LABELS: Record<string, string> = {
  pending: "Queued...",
  analyzing_audio: "Analyzing audio",
  classifying_emotion: "Classifying emotion",
  captioning_images: "Captioning images",
  matching: "Matching images to music",
  rendering: "Rendering video",
  encoding: "Encoding video",
  done: "Complete!",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function ProgressBar({ progress, onCancel }: Props) {
  if (!progress) return null;

  const pct = Math.round(progress.progress);
  const label = STEP_LABELS[progress.status] || progress.current_step;
  const isTerminal =
    progress.status === "done" ||
    progress.status === "failed" ||
    progress.status === "cancelled";
  const isFailed = progress.status === "failed";

  return (
    <div style={{ width: "100%" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 8,
          alignItems: "center",
        }}
      >
        <span style={{ fontWeight: 500 }}>{label}</span>
        <span style={{ color: "#888", fontSize: 14 }}>
          {pct}%
          {!isTerminal && onCancel && (
            <button
              onClick={onCancel}
              style={{
                marginLeft: 12,
                padding: "2px 8px",
                fontSize: 12,
                border: "1px solid #666",
                borderRadius: 4,
                background: "transparent",
                color: "#ccc",
                cursor: "pointer",
              }}
            >
              Cancel
            </button>
          )}
        </span>
      </div>

      <div
        style={{
          height: 8,
          background: "#333",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: isFailed ? "#e74c3c" : "#6c5ce7",
            borderRadius: 4,
            transition: "width 0.3s ease",
          }}
        />
      </div>

      {progress.detail && (
        <p
          style={{
            color: isFailed ? "#e74c3c" : "#888",
            fontSize: 13,
            marginTop: 6,
          }}
        >
          {progress.detail}
        </p>
      )}
    </div>
  );
}
