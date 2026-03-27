import { useCallback, useEffect, useState } from "react";
import {
  clearXhsCookies,
  clearXhsStyleProfile,
  getXhsCookieStatus,
  getXhsStyleProfile,
  saveXhsCookies,
  scanXhsStyleProfile,
} from "../api/client";
import { useI18n } from "../i18n";
import type { XhsCookieStatus, XhsStyleProfile } from "../types";

interface Props {
  onStatusChange?: (connected: boolean) => void;
}

export function XhsConfig({ onStatusChange }: Props) {
  const { t } = useI18n();
  const [status, setStatus] = useState<XhsCookieStatus | null>(null);
  const [showInput, setShowInput] = useState(false);
  const [cookie, setCookie] = useState("");
  const [saving, setSaving] = useState(false);
  const [styleProfile, setStyleProfile] = useState<XhsStyleProfile | null>(null);
  const [scanning, setScanning] = useState(false);
  const [styleError, setStyleError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await getXhsCookieStatus();
      setStatus(s);
      onStatusChange?.(s.connected);
    } catch {
      setStatus({ connected: false, username: null, user_id: null, expired: false, error: null });
      onStatusChange?.(false);
    }
  }, [onStatusChange]);

  useEffect(() => {
    fetchStatus();
    // Load cached style profile
    getXhsStyleProfile().then(setStyleProfile).catch(() => {});
  }, [fetchStatus]);

  const handleScanStyle = useCallback(async (force = false) => {
    setScanning(true);
    setStyleError(null);
    try {
      const profile = await scanXhsStyleProfile(force);
      setStyleProfile(profile);
    } catch (e: any) {
      setStyleError(e.message || "Unknown error");
    } finally {
      setScanning(false);
    }
  }, []);

  const handleClearStyle = useCallback(async () => {
    await clearXhsStyleProfile();
    setStyleProfile(null);
    setStyleError(null);
  }, []);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const s = await saveXhsCookies(cookie);
      setStatus(s);
      onStatusChange?.(s.connected);
      if (s.connected) {
        setShowInput(false);
        setCookie("");
      }
    } catch (e: any) {
      setStatus({
        connected: false,
        username: null,
        user_id: null,
        expired: false,
        error: e.message,
      });
      onStatusChange?.(false);
    } finally {
      setSaving(false);
    }
  }, [cookie, onStatusChange]);

  const handleDisconnect = useCallback(async () => {
    await clearXhsCookies();
    setStatus({ connected: false, username: null, user_id: null, expired: false, error: null });
    onStatusChange?.(false);
  }, [onStatusChange]);

  const connected = status?.connected ?? false;

  return (
    <div
      style={{
        padding: 16,
        background: "#1a1a2e",
        border: "1px solid #333",
        borderRadius: 12,
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: connected ? "#2ecc71" : status?.expired ? "#f39c12" : "#e74c3c",
            flexShrink: 0,
          }}
        />
        <span style={{ fontWeight: 600, fontSize: 15, flex: 1 }}>
          {t.xhsAccount}
          {connected && status?.username && (
            <span style={{ fontWeight: 400, color: "#888", marginLeft: 8 }}>
              {status.username}
            </span>
          )}
        </span>
        {connected ? (
          <button
            onClick={handleDisconnect}
            style={{
              padding: "5px 14px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid #e74c3c",
              background: "transparent",
              color: "#e74c3c",
              cursor: "pointer",
            }}
          >
            {t.xhsDisconnect}
          </button>
        ) : !showInput ? (
          <button
            onClick={() => setShowInput(true)}
            style={{
              padding: "5px 14px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid #6c5ce7",
              background: "#6c5ce7",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            {t.xhsConnect}
          </button>
        ) : (
          <button
            onClick={() => {
              setShowInput(false);
              setCookie("");
            }}
            style={{
              padding: "5px 14px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid #555",
              background: "transparent",
              color: "#888",
              cursor: "pointer",
            }}
          >
            {t.xhsCancel}
          </button>
        )}
      </div>

      {/* Error / expired message */}
      {status?.error && !showInput && (
        <p style={{ color: status.expired ? "#f39c12" : "#e74c3c", fontSize: 13, margin: "8px 0 0" }}>
          {status.expired ? t.xhsExpired : status.error}
        </p>
      )}

      {/* Cookie input */}
      {showInput && (
        <div style={{ marginTop: 12 }}>
          <p style={{ color: "#888", fontSize: 12, margin: "0 0 8px" }}>
            {t.xhsCookieHint}
          </p>
          <textarea
            value={cookie}
            onChange={(e) => setCookie(e.target.value)}
            placeholder="a1=...; web_session=...; ..."
            rows={3}
            style={{
              width: "100%",
              background: "#111",
              border: "1px solid #444",
              borderRadius: 8,
              color: "#eee",
              padding: 10,
              fontSize: 13,
              fontFamily: "monospace",
              resize: "vertical",
              boxSizing: "border-box",
            }}
          />
          <div style={{ textAlign: "right", marginTop: 8 }}>
            <button
              onClick={handleSave}
              disabled={saving || !cookie.trim()}
              style={{
                padding: "6px 18px",
                fontSize: 13,
                fontWeight: 600,
                borderRadius: 6,
                border: "none",
                background: saving || !cookie.trim() ? "#444" : "#6c5ce7",
                color: "#fff",
                cursor: saving || !cookie.trim() ? "not-allowed" : "pointer",
              }}
            >
              {saving ? t.xhsConnecting : t.xhsSaveCookies}
            </button>
          </div>
          {status?.error && (
            <p style={{ color: "#e74c3c", fontSize: 13, margin: "8px 0 0" }}>
              {status.error}
            </p>
          )}
        </div>
      )}

      {/* Style Scanner */}
      {connected && (
        <div style={{ marginTop: 12, borderTop: "1px solid #333", paddingTop: 12 }}>
          {styleProfile && !styleProfile.error ? (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ color: "#2ecc71", fontSize: 13, fontWeight: 600, flex: 1 }}>
                  {t.xhsStyleActive}
                </span>
                <button
                  onClick={() => handleScanStyle(true)}
                  disabled={scanning}
                  style={{
                    padding: "3px 10px",
                    fontSize: 12,
                    borderRadius: 4,
                    border: "1px solid #6c5ce7",
                    background: "transparent",
                    color: "#6c5ce7",
                    cursor: scanning ? "not-allowed" : "pointer",
                  }}
                >
                  {scanning ? t.xhsScanning : t.xhsRescan}
                </button>
                <button
                  onClick={handleClearStyle}
                  style={{
                    padding: "3px 10px",
                    fontSize: 12,
                    borderRadius: 4,
                    border: "1px solid #555",
                    background: "transparent",
                    color: "#888",
                    cursor: "pointer",
                  }}
                >
                  {t.xhsClearStyle}
                </button>
              </div>
              <p style={{ color: "#aaa", fontSize: 12, margin: 0, lineHeight: 1.5 }}>
                {styleProfile.overall_summary}
              </p>
              {styleProfile.scraped_at && (
                <p style={{ color: "#666", fontSize: 11, margin: "4px 0 0" }}>
                  {new Date(styleProfile.scraped_at).toLocaleString()}
                </p>
              )}
            </div>
          ) : (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <button
                onClick={() => handleScanStyle(false)}
                disabled={scanning}
                style={{
                  padding: "5px 14px",
                  fontSize: 13,
                  borderRadius: 6,
                  border: "1px solid #f39c12",
                  background: scanning ? "transparent" : "rgba(243,156,18,0.15)",
                  color: "#f39c12",
                  cursor: scanning ? "not-allowed" : "pointer",
                }}
              >
                {scanning ? t.xhsScanning : t.xhsScanStyle}
              </button>
            </div>
          )}
          {styleError && (
            <p style={{ color: "#e74c3c", fontSize: 12, margin: "8px 0 0" }}>
              {t.xhsStyleError(styleError)}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
