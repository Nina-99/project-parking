/** Query hooks. They talk to the port, never to fetch or a URL. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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

export function useVehicle(id: string | undefined) {
  const api = useApi();
  return useQuery({
    queryKey: ["vehicle", id],
    queryFn: () => api.getVehicle(id as string),
    enabled: Boolean(id),
  });
}

export function useCameras() {
  const api = useApi();
  return useQuery({
    queryKey: ["cameras"],
    queryFn: () => api.getCameras(),
  });
}

export function useModelStats(days = 7) {
  const api = useApi();
  return useQuery({
    queryKey: ["model-stats", days],
    queryFn: () => api.getModelStats(days),
  });
}

export function useCloseSession() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      api.closeSession(id, reason),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["occupancy"] });
      void queryClient.invalidateQueries({ queryKey: ["active-sessions"] });
      void queryClient.invalidateQueries({ queryKey: ["history"] });
      void queryClient.invalidateQueries({ queryKey: ["vehicle"] });
    },
  });
}
