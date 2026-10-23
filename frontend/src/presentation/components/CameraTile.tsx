import { Camera } from "lucide-react";
import { formatDateTime } from "../../domain/incident";
import type { CameraReading } from "../../domain/types";

export function CameraTile({ camera }: { camera: CameraReading }) {
  return (
    <div className="card flex items-start justify-between gap-3">
      <div className="flex items-start gap-3">
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center ${
            camera.online ? "bg-green-100" : "bg-red-100"
          }`}
        >
          <Camera
            className={`w-5 h-5 ${camera.online ? "text-green-700" : "text-red-700"}`}
          />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-900">{camera.camera_id}</p>
          <p className="text-xs text-gray-500">{camera.gate ?? "sin portón"}</p>
          <p className="text-xs text-gray-400">
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
              ? "bg-green-100 text-green-800"
              : "bg-red-100 text-red-800"
          }`}
        >
          {camera.online ? "En línea" : "Sin señal"}
        </span>
        {camera.dropped_frames > 0 && (
          <p className="text-xs text-amber-600 mt-1">
            {camera.dropped_frames} descartados
          </p>
        )}
      </div>
    </div>
  );
}
