import Supervise from "../features/Supervise/Supervise";
import Blacklist from "../features/Blacklist/Blacklist";

import ErrorPage from "../components/ErrorPage/ErrorPage";
import Redirect404 from "../components/ErrorPage/Redirect404";

const publicRoutes = [
  { path: "/", component: Supervise },
  // failed => navigate to error tabs
  { path: "/404", component: ErrorPage },
  { path: "*", component: Redirect404 },
];

const privateRoutes = [
  {
    path: "/Supervise",
    component: Supervise,
  },
  {
    path: "/Blacklist",
    component: Blacklist,
  },
];

export { publicRoutes, privateRoutes };
