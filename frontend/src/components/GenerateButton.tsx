import { useI18n } from "../i18n";

interface Props {
  canGenerate: boolean;
  loading: boolean;
  onClick: () => void;
}

export function GenerateButton({ canGenerate, loading, onClick }: Props) {
  const { t } = useI18n();

  return (
    <button
      onClick={onClick}
      disabled={!canGenerate || loading}
      style={{
        padding: "12px 32px",
        fontSize: 18,
        fontWeight: 600,
        borderRadius: 8,
        border: "none",
        background: canGenerate && !loading ? "#6c5ce7" : "#444",
        color: "#fff",
        cursor: canGenerate && !loading ? "pointer" : "not-allowed",
        transition: "background 0.2s",
      }}
    >
      {loading ? t.btnStarting : t.btnGenerate}
    </button>
  );
}
