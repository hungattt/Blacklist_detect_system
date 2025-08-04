import { useEffect, useRef } from "react";
import { WSContext } from "./Contexts";

interface WSContextType {
  ws: React.RefObject<WebSocket | null>;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
}

interface WSContextProviderProps {
  children: React.ReactNode;
}

const WSContextProvider: React.FC<WSContextProviderProps> = ({ children }) => {
  const ws = useRef<WebSocket | null>(null);

  function setupWebSocket(ws: WebSocket) {
    ws.onopen = () => {
      // alert("Đã kết nối websocket");
    };
    ws.onclose = () => {
      /*alert("Đã bị ngắt kết nối websocket")*/
    };
    // ws.onmessage = (event) => { ...handle messages... };
  }

  // useEffect(() => {
  //   // Clean up on unmount instead
  //   return () => {
  //     if (ws.current) {
  //       ws.current.close();
  //     }
  //   };
  // }, []);

  const connect = async (): Promise<void> => {
    const url = `${import.meta.env.VITE_WS_URL}`;
    ws.current = new WebSocket(url);
    if (ws.current) {
      setupWebSocket(ws.current);
    }
  };

  const disconnect = async (): Promise<void> => {
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
  };

  // Context data
  const wsContextData: WSContextType = { ws, connect, disconnect };

  // Return provider
  return (
    <WSContext.Provider value={wsContextData}>{children}</WSContext.Provider>
  );
};

export default WSContextProvider;
