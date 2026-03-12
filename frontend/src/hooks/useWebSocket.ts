import { useCallback, useEffect, useRef, useState } from "react";
import type { ProgressMessage } from "../types";

const TERMINAL = new Set(["done", "failed", "cancelled"]);

export function useWebSocket(taskId: string | null) {
  const [progress, setProgress] = useState<ProgressMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const terminalRef = useRef(false);

  useEffect(() => {
    if (!taskId) return;

    terminalRef.current = false;
    setProgress(null);

    let reconnectTimer: ReturnType<typeof setTimeout>;
    let disposed = false;

    function connect() {
      if (disposed || terminalRef.current) return;

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${window.location.host}/ws/progress/${taskId}`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg: ProgressMessage = JSON.parse(event.data);
          setProgress(msg);
          if (TERMINAL.has(msg.status)) {
            terminalRef.current = true;
            ws.close();
          }
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        if (!disposed && !terminalRef.current) {
          reconnectTimer = setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      disposed = true;
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [taskId]);

  const cancel = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: "cancel" }));
  }, []);

  return { progress, cancel };
}
