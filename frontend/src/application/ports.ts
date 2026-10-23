/** What the UI needs from the outside world. Adapters implement these. */
import type {
  ActiveSession,
  DailyStat,
  Occupancy,
  SessionHistory,
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
}

export type MessageHandler = (message: WsMessage) => void;

export interface EventStream {
  connect(): void;
  disconnect(): void;
  onMessage(handler: MessageHandler): () => void;
}
