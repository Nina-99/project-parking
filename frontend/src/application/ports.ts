/** What the UI needs from the outside world. Adapters implement these. */
import type {
  ActiveSession,
  CamerasResponse,
  CloseSessionResult,
  DailyStat,
  ModelStats,
  Occupancy,
  SessionHistory,
  VehicleDetail,
  VehicleSearchHit,
  VehicleSearchQuery,
  WsMessage,
} from "../domain/types";

export interface ApiPort {
  getActiveSessions(): Promise<ActiveSession[]>;
  getHistory(limit?: number, offset?: number): Promise<SessionHistory[]>;
  getOccupancy(): Promise<Occupancy>;
  getDailyStats(days?: number): Promise<DailyStat[]>;
  searchVehicles(query: VehicleSearchQuery): Promise<VehicleSearchHit[]>;
  getVehicle(id: string): Promise<VehicleDetail>;
  getCameras(): Promise<CamerasResponse>;
  getModelStats(days?: number): Promise<ModelStats>;
  closeSession(id: string, reason: string): Promise<CloseSessionResult>;
}

export type MessageHandler = (message: WsMessage) => void;

export interface EventStream {
  connect(): void;
  disconnect(): void;
  onMessage(handler: MessageHandler): () => void;
}

export interface AuthPort {
  /** The logged-in operator, or a rejection when there is no session. */
  me(): Promise<string>;
  login(username: string, password: string): Promise<string>;
  logout(): Promise<void>;
}
