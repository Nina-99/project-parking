import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AppProviders } from "./composition";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // The socket is the fast path; polling is the fallback when it is down.
      refetchInterval: 5000,
      staleTime: 3000,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <AppProviders>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </AppProviders>
    </QueryClientProvider>
  </StrictMode>
);
