// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { Camera } from "lucide-react";
import { formatDateTime } from "../../domain/incident";
import type { CameraReading } from "../../domain/types";

export function CameraTile({ camera }: { camera: CameraReading }) {
  return (
    <div className="card card-hover flex items-start justify-between gap-3">
      <div className="flex items-start gap-3">
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center ${
            camera.online ? "bg-live/10" : "bg-danger/10"
          }`}
        >
          <Camera
            className={`w-5 h-5 ${camera.online ? "text-live" : "text-danger"}`}
          />
        </div>
        <div>
          <p className="text-sm font-medium text-ink">{camera.camera_id}</p>
          <p className="text-xs text-muted">{camera.gate ?? "sin portón"}</p>
          <p className="text-xs text-subtle">
            {camera.last_seen
              ? `Última señal: ${formatDateTime(camera.last_seen)}`
              : "Nunca reportó"}
          </p>
        </div>
      </div>
      <div className="text-right">
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
            camera.online
              ? "bg-live/10 text-live"
              : "bg-danger/10 text-danger"
          }`}
        >
          {camera.online ? "En línea" : "Sin señal"}
        </span>
        {camera.dropped_frames > 0 && (
          <p className="text-xs text-warn mt-1">
            {camera.dropped_frames} descartados
          </p>
        )}
      </div>
    </div>
  );
}
