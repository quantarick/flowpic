import { useCallback, useRef } from "react";

interface Props {
  images: string[];
  onUpload: (files: File[]) => void;
  disabled?: boolean;
}

export function ImageUploader({ images, onUpload, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (disabled) return;
      const files = Array.from(e.dataTransfer.files).filter((f) =>
        f.type.startsWith("image/")
      );
      if (files.length) onUpload(files);
    },
    [onUpload, disabled]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      if (files.length) onUpload(files);
      e.target.value = "";
    },
    [onUpload]
  );

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
      {images.length === 0 ? (
        <div>
          <p style={{ fontSize: 18, margin: "0 0 8px" }}>
            Drop images here or click to browse
          </p>
          <p style={{ color: "#888", margin: 0 }}>
            JPG, PNG, WebP - up to 100 images, 20MB each
          </p>
        </div>
      ) : (
        <div>
          <p style={{ margin: "0 0 8px" }}>
            {images.length} image{images.length !== 1 ? "s" : ""} uploaded
          </p>
          <p style={{ color: "#888", margin: 0, fontSize: 14 }}>
            Click or drop to add more
          </p>
        </div>
      )}
    </div>
  );
}
