import { createContext } from "react";

type WSContextType = {
  ws: React.RefObject<WebSocket | null>;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
};

type globalStateContextDataTypes = {
  startFly: boolean;
  setStartFly: React.Dispatch<React.SetStateAction<boolean>>;
};

export const WSContext = createContext<WSContextType | null>(null);
export const GlobalStateContext =
  createContext<globalStateContextDataTypes | null>(null);
