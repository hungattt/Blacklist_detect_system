import { useState } from "react";
import { GlobalStateContext } from "./Contexts";

interface GlobalStateContextProps {
  children: React.ReactNode;
}

type globalStateContextDataTypes = {
  startFly: boolean;
  setStartFly: React.Dispatch<React.SetStateAction<boolean>>;
};

const GlobalStateContextProvider: React.FC<GlobalStateContextProps> = ({
  children,
}) => {
  const [startFly, setStartFly] = useState(false);

  const globalStateContextData: globalStateContextDataTypes = {
    startFly,
    setStartFly,
  };
  return (
    <GlobalStateContext.Provider value={globalStateContextData}>
      {children}
    </GlobalStateContext.Provider>
  );
};

export default GlobalStateContextProvider;
