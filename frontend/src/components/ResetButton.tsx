import { useEffect, useState } from "react";
import { fetchResetPreview, resetEverything } from "../api/client";
import { useI18n } from "../i18n";
import type { ResetPreview } from "../types";

interface Props {
  /** True while a task runs. The server refuses a reset then, so the button is off. */
  disabled?: boolean;
  /** Called after the data is gone, so the page can clear what it was showing. */
  onDone: () => void;
}

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

/** The API client throws "API error 409: {"detail": "..."}". Show just the detail. */
function apiDetail(e: unknown): string {
  const message = e instanceof Error ? e.message : String(e);
  const start = message.indexOf("{");
  if (start >= 0) {
    try {
      const detail = JSON.parse(message.slice(start)).detail;
      if (typeof detail === "string") return detail;
    } catch {
      // not JSON; fall through to the raw message
    }
  }
  return message;
}

export function ResetButton({ disabled, onDone }: Props) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [preview, setPreview] = useState<ResetPreview | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const openDialog = () => {
    setOpen(true);
    setPreview(null);
    setError(null);
    fetchResetPreview()
      .then(setPreview)
      .catch((e) => setError(t.resetFailed(apiDetail(e))));
  };

  const close = () => {
    if (!working) setOpen(false);
  };

  // Escape closes the dialog, unless a delete is in flight.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !working) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, working]);

  const confirmReset = async () => {
    setWorking(true);
    setError(null);
    try {
      const result = await resetEverything();
      onDone();
      if (result.failed.length > 0) {
        setError(t.resetPartly(result.failed.length));
        setPreview(null);
      } else {
        setOpen(false);
      }
    } catch (e) {
      setError(t.resetFailed(apiDetail(e)));
    } finally {
      setWorking(false);
    }
  };

  const nothingStored =
    preview !== null && preview.projects === 0 && preview.tasks === 0;

  return (
    <>
      <button
        onClick={openDialog}
        disabled={disabled}
        title={disabled ? t.resetBusy : undefined}
        style={{
          padding: "6px 14px",
          fontSize: 13,
          borderRadius: 6,
          border: "1px solid #7a4a4a",
          background: "transparent",
          color: disabled ? "#666" : "#d99",
          cursor: disabled ? "not-allowed" : "pointer",
        }}
      >
        {t.btnReset}
      </button>

      {open && (
        <div
          onClick={close}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0, 0, 0, 0.7)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            padding: 16,
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="reset-title"
            onClick={(e) => e.stopPropagation()}
            style={{
              width: "min(460px, 100%)",
              background: "#1e1e1e",
              border: "1px solid #444",
              borderRadius: 12,
              padding: 24,
              textAlign: "left",
              color: "#eee",
            }}
          >
            <h2 id="reset-title" style={{ margin: "0 0 12px", fontSize: 20 }}>
              {t.resetTitle}
            </h2>

            {preview === null && !error && (
              <p style={{ color: "#aaa", margin: "0 0 12px" }}>{t.resetCounting}</p>
            )}

            {preview !== null && nothingStored && (
              <p style={{ margin: "0 0 12px" }}>{t.resetNothing}</p>
            )}

            {preview !== null && !nothingStored && (
              <>
                <p style={{ margin: "0 0 12px", lineHeight: 1.5 }}>
                  {t.resetSummary(
                    preview.projects,
                    preview.photos,
                    preview.songs,
                    preview.videos,
                    preview.tasks,
                    formatSize(preview.bytes)
                  )}
                </p>
                <p style={{ margin: "0 0 12px", color: "#e88", fontWeight: 600 }}>
                  {t.resetWarning}
                </p>
                <p style={{ margin: "0 0 12px", color: "#999", fontSize: 13, lineHeight: 1.5 }}>
                  {t.resetKept}
                </p>
              </>
            )}

            {error && (
              <p style={{ margin: "0 0 12px", color: "#f88", lineHeight: 1.5 }}>{error}</p>
            )}

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 20 }}>
              <button
                onClick={close}
                disabled={working}
                autoFocus
                style={{
                  padding: "8px 16px",
                  borderRadius: 6,
                  border: "1px solid #555",
                  background: "transparent",
                  color: "#ddd",
                  cursor: working ? "not-allowed" : "pointer",
                }}
              >
                {nothingStored ? t.resetClose : t.resetCancel}
              </button>
              {preview !== null && !nothingStored && (
                <button
                  onClick={confirmReset}
                  disabled={working}
                  style={{
                    padding: "8px 16px",
                    borderRadius: 6,
                    border: "none",
                    background: working ? "#733" : "#c0392b",
                    color: "#fff",
                    fontWeight: 600,
                    cursor: working ? "wait" : "pointer",
                  }}
                >
                  {working ? t.resetWorking : t.resetConfirm}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
