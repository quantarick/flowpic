import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "../i18n";
import {
  MAX_PROJECT_IMAGES,
  entriesFromDrop,
  filesFromEntries,
  filesFromFileList,
  planImport,
  type FoundFile,
  type ImportPlan,
} from "../utils/folderImport";

interface Props {
  images: string[];
  onUpload: (files: File[]) => void;
  disabled?: boolean;
  /** Set while images are being sent in batches. */
  uploadProgress?: { done: number; total: number } | null;
}

export function ImageUploader({ images, onUpload, disabled, uploadProgress }: Props) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [scanning, setScanning] = useState(false);
  const [plan, setPlan] = useState<ImportPlan | null>(null);

  // `webkitdirectory` turns a file input into a folder picker that returns
  // every file in the folder and its sub-folders. React's types do not know
  // the attribute, so it is set on the element directly.
  useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  // The project was emptied (new project or reset): drop the old import note.
  useEffect(() => {
    if (images.length === 0) setPlan(null);
  }, [images.length]);

  const importFound = useCallback(
    (found: FoundFile[]) => {
      const next = planImport(found, MAX_PROJECT_IMAGES - images.length);
      setPlan(next);
      if (next.accepted.length) onUpload(next.accepted);
    },
    [images.length, onUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (disabled) return;
      // Both must be read now; the browser clears them when this handler returns.
      const entries = entriesFromDrop(e.dataTransfer.items);
      const looseFiles = Array.from(e.dataTransfer.files);

      setScanning(true);
      (entries.length ? filesFromEntries(entries) : Promise.resolve(filesFromFileList(looseFiles)))
        .then(importFound)
        .finally(() => setScanning(false));
    },
    [disabled, importFound]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const found = filesFromFileList(e.target.files ?? []);
      e.target.value = "";
      if (found.length) importFound(found);
    },
    [importFound]
  );

  const busy = scanning || !!uploadProgress;
  const skippedSomething =
    plan !== null && (plan.unsupported > 0 || plan.tooLarge > 0 || plan.leftOut > 0);

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        border: "2px dashed #666",
        borderRadius: 12,
        padding: 24,
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        minHeight: 120,
        transition: "border-color 0.2s",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        multiple
        onChange={handleChange}
        style={{ display: "none" }}
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        onChange={handleChange}
        style={{ display: "none" }}
      />

      {images.length === 0 ? (
        <div>
          <p style={{ fontSize: 18, margin: "0 0 8px" }}>{t.imgDropHint}</p>
          <p style={{ color: "#888", margin: 0 }}>{t.imgFormatHint}</p>
        </div>
      ) : (
        <div>
          <p style={{ margin: "0 0 8px" }}>{t.imgUploaded(images.length)}</p>
          <p style={{ color: "#888", margin: 0, fontSize: 14 }}>{t.imgAddMore}</p>
        </div>
      )}

      <button
        type="button"
        disabled={disabled || busy}
        onClick={(e) => {
          e.stopPropagation(); // do not also open the file picker behind it
          folderInputRef.current?.click();
        }}
        style={{
          marginTop: 12,
          padding: "6px 14px",
          fontSize: 13,
          borderRadius: 6,
          border: "1px solid #555",
          background: "transparent",
          color: "#ccc",
          cursor: disabled || busy ? "not-allowed" : "pointer",
        }}
      >
        {t.imgPickFolder}
      </button>

      {scanning && <p style={{ color: "#aaa", margin: "12px 0 0", fontSize: 14 }}>{t.imgScanning}</p>}

      {uploadProgress && (
        <p style={{ color: "#aaa", margin: "12px 0 0", fontSize: 14 }}>
          {t.imgUploadProgress(uploadProgress.done, uploadProgress.total)}
        </p>
      )}

      {plan !== null && !busy && (
        <div style={{ margin: "12px 0 0", fontSize: 13, lineHeight: 1.6 }}>
          {plan.accepted.length > 0 ? (
            <p style={{ color: "#9c9", margin: 0 }}>{t.imgImported(plan.accepted.length, plan.folders)}</p>
          ) : (
            <p style={{ color: "#e99", margin: 0 }}>{t.imgNothingFound}</p>
          )}
          {skippedSomething && (
            <p style={{ color: "#da8", margin: 0 }}>
              {[
                plan.unsupported > 0 ? t.imgSkippedType(plan.unsupported, plan.unsupportedKinds.join(", ")) : null,
                plan.tooLarge > 0 ? t.imgSkippedLarge(plan.tooLarge) : null,
                plan.leftOut > 0 ? t.imgLeftOut(plan.leftOut, MAX_PROJECT_IMAGES) : null,
              ]
                .filter(Boolean)
                .join(" ")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
