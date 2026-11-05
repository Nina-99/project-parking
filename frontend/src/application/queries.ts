// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** Query hooks. They talk to the port, never to fetch or a URL. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  AuthorizedVehicleCreateInput,
  AuthorizedVehicleUpdateInput,
  HistoryQuery,
  ValidationDecisionInput,
  VehicleSearchQuery,
} from "../domain/types";
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

export function useHistory(query?: HistoryQuery) {
  const api = useApi();
  return useQuery({
    queryKey: [
      "history",
      query?.limit,
      query?.offset,
      query?.gate,
      query?.from_date,
      query?.to_date,
      query?.search,
      query?.status,
    ],
    queryFn: () => api.getHistory(query),
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

export function useModelStats(days = 7, enabled = true) {
  const api = useApi();
  return useQuery({
    queryKey: ["model-stats", days],
    queryFn: () => api.getModelStats(days),
    enabled,
  });
}

export function useQualityStats(days = 7) {
  const api = useApi();
  return useQuery({
    queryKey: ["quality-stats", days],
    queryFn: () => api.getQualityStats(days),
  });
}

export function useColourStats(days = 30) {
  const api = useApi();
  return useQuery({
    queryKey: ["colour-stats", days],
    queryFn: () => api.getColourStats(days),
  });
}

export function useHourlyStats(days = 7) {
  const api = useApi();
  return useQuery({
    queryKey: ["hourly-stats", days],
    queryFn: () => api.getHourlyStats(days),
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

export function usePendingValidations(hours = 8) {
  const api = useApi();
  return useQuery({
    queryKey: ["pending-validations", hours],
    queryFn: () => api.getPendingValidations(hours),
    // The socket invalidates this query on every gate event. The interval is a
    // safety net for a dropped socket, not the primary mechanism.
    refetchInterval: 30000,
  });
}

export function useSubmitValidationDecision() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      vehicleId,
      input,
    }: {
      vehicleId: string;
      input: ValidationDecisionInput;
    }) => api.submitValidationDecision(vehicleId, input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["pending-validations"] });
      void queryClient.invalidateQueries({ queryKey: ["quality-stats"] });
      void queryClient.invalidateQueries({ queryKey: ["vehicle"] });
      void queryClient.invalidateQueries({ queryKey: ["vehicle-search"] });
      void queryClient.invalidateQueries({ queryKey: ["history"] });
    },
  });
}

export function useAuthorizedVehicles(query?: {
  search?: string;
  active_only?: boolean;
}) {
  const api = useApi();
  return useQuery({
    queryKey: ["whitelist", query?.search, query?.active_only],
    queryFn: () => api.listAuthorizedVehicles(query),
  });
}

export function useCreateAuthorizedVehicle() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: AuthorizedVehicleCreateInput) =>
      api.createAuthorizedVehicle(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["whitelist"] });
    },
  });
}

export function useUpdateAuthorizedVehicle() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      input,
    }: {
      id: string;
      input: AuthorizedVehicleUpdateInput;
    }) => api.updateAuthorizedVehicle(id, input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["whitelist"] });
    },
  });
}

export function useDeleteAuthorizedVehicle() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteAuthorizedVehicle(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["whitelist"] });
    },
  });
}
