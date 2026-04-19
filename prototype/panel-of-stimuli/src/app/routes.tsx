import { createBrowserRouter } from "react-router";
import { SuccessFlow } from "./components/SuccessFlow";
import { ErrorFlow } from "./components/ErrorFlow";
import { MessageFlow } from "./components/MessageFlow";
import { LoadingFlow } from "./components/LoadingFlow";
import { Home } from "./components/Home";

export const router = createBrowserRouter([
  {
    path: "/",
    children: [
      { index: true, Component: Home },
      { path: "success", Component: SuccessFlow },
      { path: "error", Component: ErrorFlow },
      { path: "message", Component: MessageFlow },
      { path: "loading", Component: LoadingFlow },
    ],
  },
]);