interface Props {
  title: string;
  description: string;
  hashtags: string[];
  coverUrl: string | null;
}

export function XhsPreview({ title, description, hashtags, coverUrl }: Props) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        padding: "24px 0",
      }}
    >
      {/* Phone frame */}
      <div
        style={{
          width: 375,
          maxWidth: "100%",
          background: "#fff",
          borderRadius: 32,
          overflow: "hidden",
          boxShadow: "0 4px 40px rgba(0,0,0,0.45)",
          border: "6px solid #1a1a1a",
          position: "relative",
        }}
      >
        {/* Status bar */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "8px 20px 4px",
            background: "#fff",
            fontSize: 12,
            fontWeight: 600,
            color: "#000",
          }}
        >
          <span>9:41</span>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            {/* Signal bars */}
            <svg width="16" height="12" viewBox="0 0 16 12">
              <rect x="0" y="8" width="3" height="4" rx="0.5" fill="#000" />
              <rect x="4.5" y="5" width="3" height="7" rx="0.5" fill="#000" />
              <rect x="9" y="2" width="3" height="10" rx="0.5" fill="#000" />
              <rect x="13.5" y="0" width="3" height="12" rx="0.5" fill="#000" opacity="0.3" />
            </svg>
            {/* WiFi */}
            <svg width="14" height="12" viewBox="0 0 14 12" fill="#000">
              <path d="M7 10.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM3.5 8a5 5 0 017 0" fill="none" stroke="#000" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M1 5.2a8.5 8.5 0 0112 0" fill="none" stroke="#000" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            {/* Battery */}
            <svg width="24" height="12" viewBox="0 0 24 12">
              <rect x="0" y="1" width="20" height="10" rx="2" stroke="#000" strokeWidth="1" fill="none" />
              <rect x="2" y="3" width="14" height="6" rx="1" fill="#000" />
              <rect x="21" y="3.5" width="2" height="5" rx="1" fill="#000" opacity="0.4" />
            </svg>
          </div>
        </div>

        {/* XHS nav bar */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "6px 16px 10px",
            background: "#fff",
          }}
        >
          {/* Back arrow */}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#222" strokeWidth="2" strokeLinecap="round">
            <path d="M15 18l-6-6 6-6" />
          </svg>
          <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
            {/* Share */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#222" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 12v7a2 2 0 002 2h12a2 2 0 002-2v-7" />
              <polyline points="16 6 12 2 8 6" />
              <line x1="12" y1="2" x2="12" y2="15" />
            </svg>
            {/* More */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="#222">
              <circle cx="5" cy="12" r="2" />
              <circle cx="12" cy="12" r="2" />
              <circle cx="19" cy="12" r="2" />
            </svg>
          </div>
        </div>

        {/* Cover image — XHS standard 3:4 portrait */}
        {coverUrl ? (
          <img
            src={coverUrl}
            alt="cover"
            style={{
              width: "100%",
              aspectRatio: "3 / 4",
              objectFit: "cover",
              display: "block",
            }}
          />
        ) : (
          <div
            style={{
              width: "100%",
              aspectRatio: "3 / 4",
              background: "#f5f5f5",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#bbb",
              fontSize: 14,
            }}
          >
            No cover
          </div>
        )}

        {/* Content area */}
        <div style={{ padding: "14px 16px 0" }}>
          {/* Author row */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 14,
            }}
          >
            {/* Avatar */}
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: "50%",
                background: "linear-gradient(135deg, #ff2442, #ff6b81)",
                flexShrink: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff",
                fontSize: 14,
                fontWeight: 700,
              }}
            >
              F
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#333" }}>
                FlowPic
              </div>
            </div>
            {/* Follow button */}
            <button
              style={{
                padding: "5px 16px",
                fontSize: 13,
                fontWeight: 600,
                borderRadius: 20,
                border: "none",
                background: "#ff2442",
                color: "#fff",
                cursor: "default",
              }}
            >
              + 关注
            </button>
          </div>

          {/* Title */}
          <h2
            style={{
              margin: "0 0 8px",
              fontSize: 18,
              fontWeight: 700,
              color: "#222",
              lineHeight: 1.4,
              letterSpacing: "0.2px",
            }}
          >
            {title}
          </h2>

          {/* Description */}
          <p
            style={{
              margin: "0 0 10px",
              fontSize: 15,
              color: "#333",
              lineHeight: 1.8,
              whiteSpace: "pre-line",
              wordBreak: "break-word",
            }}
          >
            {description}
          </p>

          {/* Hashtags */}
          {hashtags.length > 0 && (
            <div style={{ marginBottom: 12, lineHeight: 2 }}>
              {hashtags.map((tag, i) => (
                <span
                  key={i}
                  style={{
                    color: "#2f7ddb",
                    fontSize: 15,
                    marginRight: 4,
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          )}

          {/* Date line */}
          <div
            style={{
              fontSize: 12,
              color: "#bbb",
              marginBottom: 14,
            }}
          >
            03-25 来自 FlowPic
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: "#f0f0f0" }} />

        {/* Interaction bar */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-around",
            alignItems: "center",
            padding: "12px 8px",
            background: "#fff",
          }}
        >
          <InteractionIcon
            icon={
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#666" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z" />
              </svg>
            }
            label="128"
          />
          <InteractionIcon
            icon={
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#666" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z" />
              </svg>
            }
            label="56"
          />
          <InteractionIcon
            icon={
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#666" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
              </svg>
            }
            label="32"
          />
        </div>

        {/* Comment input */}
        <div style={{ padding: "0 16px 14px" }}>
          <div
            style={{
              background: "#f5f5f5",
              borderRadius: 20,
              padding: "10px 16px",
              fontSize: 14,
              color: "#bbb",
            }}
          >
            说点什么...
          </div>
        </div>
      </div>
    </div>
  );
}

function InteractionIcon({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
      }}
    >
      {icon}
      <span style={{ fontSize: 13, color: "#666" }}>{label}</span>
    </div>
  );
}
