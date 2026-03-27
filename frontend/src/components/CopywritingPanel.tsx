import { useCallback, useEffect, useState } from "react";
import { generateCopywriting, getCopywriting, getCropUrl, getPublishedImages, publishToXhs, removePublishedImages } from "../api/client";
import { useI18n } from "../i18n";
import type { CopywritingResult, XhsPublishResult } from "../types";
import { VideoPreview } from "./VideoPreview";
import { XhsConfig } from "./XhsConfig";
import { XhsPreview } from "./XhsPreview";

interface Props {
  projectId: string | null;
  crops: string[];
  videoUrl: string | null;
  taskId: string | null;
  onProjectRefresh?: () => void;
}

export function CopywritingPanel({ projectId, crops, videoUrl, taskId, onProjectRefresh }: Props) {
  const { t } = useI18n();
  const [result, setResult] = useState<CopywritingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [selectedImages, setSelectedImages] = useState<Set<number>>(new Set());
  const [showPreview, setShowPreview] = useState(false);
  const [xhsConnected, setXhsConnected] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [publishResult, setPublishResult] = useState<XhsPublishResult | null>(null);
  const [hint, setHint] = useState("");
  const [publishedFilenames, setPublishedFilenames] = useState<Set<string>>(new Set());

  // Load cached result and published images on mount
  useEffect(() => {
    if (!projectId) return;
    getCopywriting(projectId).then(setResult).catch(() => {});
    getPublishedImages(projectId)
      .then((data) => {
        setPublishedFilenames(new Set(data.published_images.map((img) => img.crop_filename)));
      })
      .catch(() => {});
  }, [projectId]);

  const MAX_PUBLISH_IMAGES = 15;

  // Select up to max unpublished crops by default when they load
  useEffect(() => {
    if (crops.length > 0) {
      const unpublished = crops
        .map((fname, i) => ({ fname, i }))
        .filter(({ fname }) => !publishedFilenames.has(fname))
        .slice(0, MAX_PUBLISH_IMAGES)
        .map(({ i }) => i);
      setSelectedImages(new Set(unpublished));
    }
  }, [crops, publishedFilenames]);

  const handleGenerate = useCallback(async () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    try {
      const r = await generateCopywriting(projectId, hint);
      setResult(r);
    } catch (e: any) {
      setError(e.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [projectId, hint]);

  const handlePublish = useCallback(async () => {
    if (!projectId || !result) return;
    const filenames = Array.from(selectedImages)
      .sort((a, b) => a - b)
      .map((i) => crops[i])
      .filter(Boolean);
    if (filenames.length === 0) return;
    setPublishing(true);
    setPublishResult(null);
    try {
      const r = await publishToXhs(projectId, {
        title: result.title,
        description: result.description,
        hashtags: result.hashtags,
        image_filenames: filenames,
      });
      setPublishResult(r);
      // Refresh published images on success
      if (r.success && projectId) {
        getPublishedImages(projectId)
          .then((data) => {
            setPublishedFilenames(new Set(data.published_images.map((img) => img.crop_filename)));
          })
          .catch(() => {});
      }
    } catch (e: any) {
      setPublishResult({ success: false, post_url: null, note_id: null, error: e.message });
    } finally {
      setPublishing(false);
    }
  }, [projectId, result, crops, selectedImages]);

  const [cleanedCount, setCleanedCount] = useState<number | null>(null);

  const handleCleanPublished = useCallback(async () => {
    if (!projectId) return;
    try {
      const res = await removePublishedImages(projectId);
      setCleanedCount(res.removed);
      onProjectRefresh?.();
      setTimeout(() => setCleanedCount(null), 3000);
    } catch {
      // ignore
    }
  }, [projectId, onProjectRefresh]);

  const copyToClipboard = useCallback(
    (text: string, label: string) => {
      navigator.clipboard.writeText(text).then(() => {
        setCopied(label);
        setTimeout(() => setCopied(null), 1500);
      });
    },
    []
  );

  if (!projectId) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* XHS Account Config */}
      <XhsConfig onStatusChange={setXhsConnected} />

      {/* Clean up published images */}
      {publishedFilenames.size > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button
            onClick={handleCleanPublished}
            style={{
              padding: "5px 14px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid #e67e22",
              background: "rgba(230,126,34,0.15)",
              color: "#e67e22",
              cursor: "pointer",
            }}
          >
            {t.btnCleanPublished} ({publishedFilenames.size})
          </button>
          {cleanedCount !== null && (
            <span style={{ color: "#2ecc71", fontSize: 13 }}>
              {t.cleanedMsg(cleanedCount)}
            </span>
          )}
        </div>
      )}

      {/* Crop thumbnail strip */}
      {crops.length > 0 && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 600, margin: "0 0 8px" }}>
            {t.publishCrops}
          </h3>
          <div
            style={{
              display: "flex",
              gap: 6,
              overflowX: "auto",
              paddingBottom: 4,
            }}
          >
            {crops.map((filename) => (
              <img
                key={filename}
                src={getCropUrl(projectId, filename)}
                alt={filename}
                style={{
                  width: 80,
                  height: 80,
                  objectFit: "cover",
                  borderRadius: 6,
                  flexShrink: 0,
                }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Video player */}
      {videoUrl && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 600, margin: "0 0 8px" }}>
            {t.publishVideo}
          </h3>
          <VideoPreview videoUrl={videoUrl} taskId={taskId} />
        </div>
      )}

      {/* Hint input */}
      <div>
        <h3 style={{ fontSize: 15, fontWeight: 600, margin: "0 0 8px" }}>
          {t.copyHint}
        </h3>
        <textarea
          value={hint}
          onChange={(e) => setHint(e.target.value)}
          placeholder={t.copyHintPlaceholder}
          rows={2}
          style={textareaStyle}
        />
      </div>

      {/* Generate button */}
      <div style={{ textAlign: "center" }}>
        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            padding: "12px 24px",
            fontSize: 16,
            fontWeight: 600,
            borderRadius: 8,
            border: loading ? "1px solid #444" : "1px solid #e74c3c",
            background: loading ? "transparent" : "#e74c3c",
            color: loading ? "#666" : "#fff",
            cursor: loading ? "not-allowed" : "pointer",
            transition: "all 0.2s",
          }}
        >
          {loading ? t.btnGeneratingCopy : t.btnGenerateCopy}
        </button>
      </div>

      {error && (
        <div
          style={{
            padding: 12,
            background: "rgba(231,76,60,0.15)",
            border: "1px solid #e74c3c",
            borderRadius: 8,
            color: "#e74c3c",
          }}
        >
          {t.copyError(error)}
        </div>
      )}

      {!result && !loading && !error && (
        <p style={{ color: "#888", textAlign: "center", fontSize: 14, margin: 0 }}>
          {t.copyEmpty}
        </p>
      )}

      {result && (
        <>
          {/* Action buttons: Copy All + Preview toggle */}
          <div style={{ display: "flex", justifyContent: "center", gap: 12 }}>
            <button
              onClick={() => {
                const full = `${result.title}\n\n${result.description}\n\n${result.hashtags.join(" ")}`;
                copyToClipboard(full, "all");
              }}
              style={{
                padding: "8px 20px",
                fontSize: 14,
                fontWeight: 600,
                borderRadius: 8,
                border: "1px solid #6c5ce7",
                background: copied === "all" ? "#6c5ce7" : "transparent",
                color: copied === "all" ? "#fff" : "#6c5ce7",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              {copied === "all" ? t.copiedMsg : `${t.btnCopy} All`}
            </button>
            <button
              onClick={() => setShowPreview((p) => !p)}
              style={{
                padding: "8px 20px",
                fontSize: 14,
                fontWeight: 600,
                borderRadius: 8,
                border: "1px solid #e74c3c",
                background: showPreview ? "#e74c3c" : "transparent",
                color: showPreview ? "#fff" : "#e74c3c",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              {showPreview ? t.btnEdit : t.btnPreview}
            </button>
          </div>

          {showPreview ? (
            <XhsPreview
              title={result.title}
              description={result.description}
              hashtags={result.hashtags}
              coverUrl={
                selectedImages.size > 0
                  ? getCropUrl(projectId, crops[Math.min(...selectedImages)])
                  : null
              }
            />
          ) : (
            <>
              {/* Title */}
              <Field
                label={t.copyTitle}
                copyLabel="title"
                copied={copied}
                onCopy={() => copyToClipboard(result.title, "title")}
                btnCopy={t.btnCopy}
                copiedMsg={t.copiedMsg}
              >
                <textarea
                  value={result.title}
                  onChange={(e) => setResult({ ...result, title: e.target.value })}
                  rows={2}
                  style={textareaStyle}
                />
              </Field>

              {/* Description */}
              <Field
                label={t.copyDesc}
                copyLabel="desc"
                copied={copied}
                onCopy={() => copyToClipboard(result.description, "desc")}
                btnCopy={t.btnCopy}
                copiedMsg={t.copiedMsg}
              >
                <textarea
                  value={result.description}
                  onChange={(e) => setResult({ ...result, description: e.target.value })}
                  rows={6}
                  style={textareaStyle}
                />
              </Field>

              {/* Hashtags */}
              <Field
                label={t.copyHashtags}
                copyLabel="tags"
                copied={copied}
                onCopy={() => copyToClipboard(result.hashtags.join(" "), "tags")}
                btnCopy={t.btnCopy}
                copiedMsg={t.copiedMsg}
              >
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {result.hashtags.map((tag, i) => (
                    <span
                      key={i}
                      style={{
                        padding: "4px 12px",
                        background: "rgba(108,92,231,0.2)",
                        border: "1px solid #6c5ce7",
                        borderRadius: 16,
                        fontSize: 13,
                        color: "#b8b0f0",
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </Field>

              {/* Image selector (multi-select) */}
              {crops.length > 0 && (
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <h3 style={{ fontSize: 15, fontWeight: 600, margin: 0 }}>
                      {t.copyCover} ({selectedImages.size}/{Math.min(crops.length, MAX_PUBLISH_IMAGES)})
                    </h3>
                    <button
                      onClick={() =>
                        setSelectedImages((prev) => {
                          if (prev.size > 0) return new Set();
                          const unpublished = crops
                            .map((fname, i) => ({ fname, i }))
                            .filter(({ fname }) => !publishedFilenames.has(fname))
                            .slice(0, MAX_PUBLISH_IMAGES)
                            .map(({ i }) => i);
                          return new Set(unpublished.length > 0 ? unpublished : crops.slice(0, MAX_PUBLISH_IMAGES).map((_, i) => i));
                        })
                      }
                      style={{
                        padding: "3px 10px",
                        fontSize: 12,
                        borderRadius: 4,
                        border: "1px solid #555",
                        background: "transparent",
                        color: "#aaa",
                        cursor: "pointer",
                      }}
                    >
                      {selectedImages.size === crops.length ? t.btnDeselectAll || "Deselect All" : t.btnSelectAll || "Select All"}
                    </button>
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
                      gap: 8,
                    }}
                  >
                    {crops.map((filename, i) => {
                      const selected = selectedImages.has(i);
                      const isPublished = publishedFilenames.has(filename);
                      return (
                        <div
                          key={filename}
                          onClick={() =>
                            setSelectedImages((prev) => {
                              const next = new Set(prev);
                              if (next.has(i)) next.delete(i);
                              else if (next.size < MAX_PUBLISH_IMAGES) next.add(i);
                              return next;
                            })
                          }
                          style={{
                            borderRadius: 8,
                            overflow: "hidden",
                            border: selected
                              ? "3px solid #e74c3c"
                              : "3px solid transparent",
                            cursor: "pointer",
                            opacity: isPublished && !selected ? 0.35 : selected ? 1 : 0.4,
                            transition: "all 0.15s",
                            position: "relative",
                            filter: isPublished ? "grayscale(60%)" : "none",
                          }}
                        >
                          <img
                            src={getCropUrl(projectId, filename)}
                            alt={filename}
                            style={{
                              width: "100%",
                              aspectRatio: "1",
                              objectFit: "cover",
                              display: "block",
                            }}
                          />
                          {selected && (
                            <div
                              style={{
                                position: "absolute",
                                top: 4,
                                right: 4,
                                width: 22,
                                height: 22,
                                borderRadius: "50%",
                                background: "#e74c3c",
                                color: "#fff",
                                fontSize: 13,
                                fontWeight: 700,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                              }}
                            >
                              {Array.from(selectedImages).sort((a, b) => a - b).indexOf(i) + 1}
                            </div>
                          )}
                          {isPublished && (
                            <div
                              style={{
                                position: "absolute",
                                bottom: 4,
                                left: 4,
                                padding: "2px 6px",
                                borderRadius: 4,
                                background: "rgba(46,204,113,0.85)",
                                color: "#fff",
                                fontSize: 10,
                                fontWeight: 700,
                                lineHeight: 1.2,
                              }}
                            >
                              {t.publishedBadge}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}

          {/* Publish to XHS button */}
          {xhsConnected && crops.length > 0 && (
            <div style={{ textAlign: "center", marginTop: 4 }}>
              <button
                onClick={handlePublish}
                disabled={publishing || selectedImages.size === 0}
                style={{
                  padding: "12px 32px",
                  fontSize: 16,
                  fontWeight: 600,
                  borderRadius: 8,
                  border: "none",
                  background: publishing || selectedImages.size === 0 ? "#444" : "#ff2442",
                  color: "#fff",
                  cursor: publishing || selectedImages.size === 0 ? "not-allowed" : "pointer",
                  transition: "background 0.2s",
                }}
              >
                {publishing
                  ? t.xhsPublishing
                  : `${t.xhsBtnPublish} (${selectedImages.size} ${selectedImages.size === 1 ? "pic" : "pics"})`}
              </button>
            </div>
          )}

          {/* Publish result banner */}
          {publishResult && (
            <div
              style={{
                padding: 12,
                borderRadius: 8,
                background: publishResult.success
                  ? "rgba(46,204,113,0.15)"
                  : "rgba(231,76,60,0.15)",
                border: `1px solid ${publishResult.success ? "#2ecc71" : "#e74c3c"}`,
                color: publishResult.success ? "#2ecc71" : "#e74c3c",
                textAlign: "center",
              }}
            >
              {publishResult.success ? (
                <>
                  {t.xhsPublishSuccess}{" "}
                  {publishResult.post_url && (
                    <a
                      href={publishResult.post_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: "#2ecc71", fontWeight: 600 }}
                    >
                      {t.xhsViewPost}
                    </a>
                  )}
                </>
              ) : (
                t.xhsPublishFailed(publishResult.error || "Unknown error")
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ---- helpers ---- */

function Field({
  label,
  copyLabel,
  copied,
  onCopy,
  btnCopy,
  copiedMsg,
  children,
}: {
  label: string;
  copyLabel: string;
  copied: string | null;
  onCopy: () => void;
  btnCopy: string;
  copiedMsg: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 6,
        }}
      >
        <h3 style={{ fontSize: 15, fontWeight: 600, margin: 0 }}>{label}</h3>
        <button
          onClick={onCopy}
          style={{
            padding: "3px 10px",
            fontSize: 12,
            borderRadius: 4,
            border: "1px solid #555",
            background: copied === copyLabel ? "#6c5ce7" : "transparent",
            color: copied === copyLabel ? "#fff" : "#aaa",
            cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          {copied === copyLabel ? copiedMsg : btnCopy}
        </button>
      </div>
      {children}
    </div>
  );
}

const textareaStyle: React.CSSProperties = {
  width: "100%",
  background: "#1a1a2e",
  border: "1px solid #333",
  borderRadius: 8,
  color: "#eee",
  padding: 12,
  fontSize: 14,
  lineHeight: 1.6,
  resize: "vertical",
  fontFamily: "inherit",
  boxSizing: "border-box",
};
