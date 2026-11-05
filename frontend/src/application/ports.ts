// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** What the UI needs from the outside world. Adapters implement these. */
import type {
  ActiveSession,
  AuthorizedVehicle,
  AuthorizedVehicleCreateInput,
  AuthorizedVehicleUpdateInput,
  CameraCreateInput,
  CamerasResponse,
  CameraUpdateInput,
  CloseSessionResult,
  ColourStats,
  DailyStat,
  HourlyStats,
  ManagedUser,
  ModelStats,
  Occupancy,
  HistoryQuery,
  PaginatedHistory,
  PendingValidation,
  QualityStats,
  SessionUser,
  ValidationDecisionInput,
  ValidationDecisionResult,
  VehicleDetail,
  VehicleSearchHit,
  UserRole,
  VehicleSearchQuery,
  WsMessage,
} from "../domain/types";

export interface ApiPort {
  getActiveSessions(): Promise<ActiveSession[]>;
  getHistory(query?: HistoryQuery): Promise<PaginatedHistory>;
  getOccupancy(): Promise<Occupancy>;
  getDailyStats(days?: number): Promise<DailyStat[]>;
  searchVehicles(query: VehicleSearchQuery): Promise<VehicleSearchHit[]>;
  getVehicle(id: string): Promise<VehicleDetail>;
  getCameras(): Promise<CamerasResponse>;
  createCamera(input: CameraCreateInput): Promise<void>;
  updateCamera(cameraId: string, input: CameraUpdateInput): Promise<void>;
  deleteCamera(cameraId: string): Promise<void>;
  getModelStats(days?: number): Promise<ModelStats>;
  closeSession(id: string, reason: string): Promise<CloseSessionResult>;

  getPendingValidations(hours?: number): Promise<PendingValidation[]>;
  submitValidationDecision(
    vehicleId: string,
    input: ValidationDecisionInput,
  ): Promise<ValidationDecisionResult>;
  getQualityStats(days?: number): Promise<QualityStats>;
  getColourStats(days?: number): Promise<ColourStats>;
  getHourlyStats(days?: number): Promise<HourlyStats>;

  listAuthorizedVehicles(query?: {
    search?: string;
    active_only?: boolean;
  }): Promise<AuthorizedVehicle[]>;
  createAuthorizedVehicle(
    input: AuthorizedVehicleCreateInput,
  ): Promise<AuthorizedVehicle>;
  updateAuthorizedVehicle(
    id: string,
    input: AuthorizedVehicleUpdateInput,
  ): Promise<AuthorizedVehicle>;
  deleteAuthorizedVehicle(id: string): Promise<void>;
  checkAuthorization(
    licensePlate: string,
  ): Promise<{ authorized: boolean; vehicle: AuthorizedVehicle | null }>;

  listUsers(): Promise<ManagedUser[]>;
  createUser(input: {
    username: string;
    password: string;
    role: UserRole;
  }): Promise<ManagedUser>;
  updateUser(
    id: string,
    input: { role?: UserRole; is_active?: boolean; password?: string },
  ): Promise<ManagedUser>;
  deleteUser(id: string): Promise<void>;
}

export type MessageHandler = (message: WsMessage) => void;

export interface EventStream {
  connect(): void;
  disconnect(): void;
  onMessage(handler: MessageHandler): () => void;
}

export interface AuthPort {
  /** The logged-in user and its role, or a rejection when there is no session. */
  me(): Promise<SessionUser>;
  login(username: string, password: string): Promise<SessionUser>;
  logout(): Promise<void>;
}
