import { useContext, useState } from "react";
import { Link } from "react-router-dom";
import { GlobalStateContext } from "../../utils/context/Contexts";

import {
  AppBar,
  Box,
  Toolbar,
  IconButton,
  Typography,
  Menu,
  Container,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";

import Logo_white from "../../assets/images/camera_logo_white.png";

const pages = [
  {
    index: 0,
    ten_navbar: "Giám sát",
    url: "/Supervise",
  },
  {
    index: 1,
    ten_navbar: "Danh sách đen",
    url: "/BlackList",
  },
];

const Navbar = () => {
  const [anchorElNav, setAnchorElNav] = useState<null | HTMLElement>(null);

  const globalStateContext = useContext(GlobalStateContext);
  const startFly = globalStateContext?.startFly;

  const handleOpenNavMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorElNav(event.currentTarget);
  };

  const handleCloseNavMenu = () => {
    setAnchorElNav(null);
  };
  return (
    <>
      <AppBar position="static">
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            {/* responsive nav */}
            <img
              className="w-12 h-12 max-sm:hidden sm:hidden md:flex md:mr-1"
              src={Logo_white}
              alt="logo"
            />
            <Typography
              variant="h6"
              noWrap
              component="a"
              sx={{
                ml: 2,
                mr: 2,
                display: { xs: "none", md: "flex" },
                fontFamily: "monospace",
                fontWeight: 700,
                letterSpacing: ".3rem",
                color: "inherit",
                textDecoration: "none",
              }}
            >
              T2K
            </Typography>

            <Box sx={{ flexGrow: 1, display: { xs: "flex", md: "none" } }}>
              <IconButton
                size="large"
                aria-label="account of current user"
                aria-controls="menu-appbar"
                aria-haspopup="true"
                onClick={handleOpenNavMenu}
                color="inherit"
              >
                <MenuIcon />
              </IconButton>

              <Menu
                id="menu-appbar"
                anchorEl={anchorElNav}
                anchorOrigin={{
                  vertical: "bottom",
                  horizontal: "left",
                }}
                keepMounted
                transformOrigin={{
                  vertical: "top",
                  horizontal: "left",
                }}
                open={Boolean(anchorElNav)}
                onClose={handleCloseNavMenu}
                sx={{
                  display: { xs: "block", md: "none" },
                }}
              >
                {pages.map((page) => (
                  <Link
                    className={`flex !p-[10px] no-underline text-black w-full 
                  h-full cursor-pointer select-none text-center text-[0.975rem] 
                  leading-[1.75] rounded-[4px] transition-all 
                  duration-[250ms] [transition-timing-function:cubic-bezier(0.4,0,0.2,1)] 
                  hover:no-underline hover:bg-[rgba(0,0,0,0.1)] ${
                    startFly
                      ? page.ten_navbar !== "Giám sát"
                        ? "pointer-events-none opacity-20"
                        : "pointer-events-none"
                      : ""
                  }`}
                    onClick={handleCloseNavMenu}
                    key={page.index}
                    to={page.url}
                  >
                    {page.ten_navbar}
                  </Link>
                ))}
              </Menu>
            </Box>
            {/* main nav */}
            <img
              className="w-12 h-12 sm:flex sm:mr-1 md:hidden"
              src={Logo_white}
              alt="logo"
            />
            <Typography
              variant="h5"
              noWrap
              component="a"
              sx={{
                ml: 2,
                mr: 2,
                display: { xs: "flex", md: "none" },
                flexGrow: 1,
                fontFamily: "monospace",
                fontWeight: 700,
                letterSpacing: ".3rem",
                color: "inherit",
                textDecoration: "none",
              }}
            >
              T2K
            </Typography>

            <Box sx={{ flexGrow: 1, display: { xs: "none", md: "flex" } }}>
              {pages.map((page) => (
                <Link
                  className={`relative box-border bg-transparent outline-none 
                        border-none m-0 cursor-pointer select-none no-underline 
                        text-center font-['Roboto'] font-medium text-[0.875rem] 
                        leading-[1.75] uppercase min-w-[64px] !px-[8px] !py-[6px] 
                        rounded-[4px] transition-all duration-[250ms] [transition-timing-function:cubic-bezier(0.4,0,0.2,1)] 
                        !ml-[32px] text-white block hover:no-underline 
                        hover:bg-[rgba(255,255,255,0.1)] active:no-underline 
                        active:bg-[rgba(255,255,255,0.1)] ${
                          startFly
                            ? page.ten_navbar !== "Giám sát"
                              ? "pointer-events-none opacity-20"
                              : "pointer-events-none"
                            : ""
                        }`}
                  key={page.index}
                  to={page.url}
                >
                  {page.ten_navbar}
                </Link>
              ))}
            </Box>
          </Toolbar>
        </Container>
      </AppBar>
    </>
  );
};

export default Navbar;
