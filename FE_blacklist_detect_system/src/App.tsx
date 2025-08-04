import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import WSContextProvider from "./utils/context/WSContext";
import GlobalStateContextProvider from "./utils/context/GlobalStateContext";
import { createTheme, ThemeProvider } from "@mui/material";
import { HashRouter, Route, Routes } from "react-router-dom";
import { publicRoutes, privateRoutes } from "./routes/routes";

import Navbar from "./components/Navbar/Navbar";

function App() {
  const theme = createTheme({
    typography: {
      fontFamily: "Roboto",
    },
  });
  return (
    <>
      <ThemeProvider theme={theme}>
        <ToastContainer
          style={{ zIndex: 99999 }}
          position="top-center"
          autoClose={2000}
          closeOnClick={true}
        />
        <GlobalStateContextProvider>
          <WSContextProvider>
            <HashRouter>
              <Navbar />
              <Routes>
                {publicRoutes.map((publicRoute, index) => {
                  return (
                    <Route
                      key={index}
                      path={publicRoute.path}
                      element={<publicRoute.component />}
                    />
                  );
                })}
                {privateRoutes.map((privateRoute, index) => {
                  return (
                    <Route
                      key={index}
                      path={privateRoute.path}
                      element={<privateRoute.component />}
                    />
                  );
                })}
              </Routes>
            </HashRouter>
          </WSContextProvider>
        </GlobalStateContextProvider>
      </ThemeProvider>
    </>
  );
}

export default App;
