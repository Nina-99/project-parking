/** Query hooks. They talk to the port, never to fetch or a URL. */
import { useQuery } from "@tanstack/react-query";
import type { VehicleSearchQuery } from "../domain/types";
import { useApi } from "./context";

export function useOccupancy() {
  const api = useApi();
  return useQuery({
    queryKey: ["occupancy"],
    queryFn: () => api.getOccupancy(),
  });
}

export function useActiveSessions() {
  const api = useApi();
  return useQuery({
    queryKey: ["active-sessions"],
    queryFn: () => api.getActiveSessions(),
  });
}

export function useDailyStats(days = 7) {
  const api = useApi();
  return useQuery({
    queryKey: ["daily-stats", days],
    queryFn: () => api.getDailyStats(days),
  });
}

export function useHistory(limit = 200) {
  const api = useApi();
  return useQuery({
    queryKey: ["history", limit],
    queryFn: () => api.getHistory(limit),
  });
}

export function useVehicleSearch(query: VehicleSearchQuery | null) {
  const api = useApi();
  return useQuery({
    queryKey: [
      "vehicle-search",
      query?.from,
      query?.to,
      query?.color,
      query?.type,
    ],
    queryFn: () => api.searchVehicles(query as VehicleSearchQuery),
    enabled: query !== null,
  });
}
