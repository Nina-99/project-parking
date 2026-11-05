// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

const SIZE = 188;
const STROKE = 14;

export function OccupancyGauge({
  active,
  capacity,
  percent,
}: {
  active: number;
  capacity: number;
  percent: number | null;
}) {
  const radius = (SIZE - STROKE) / 2;
  const circumference = 2 * Math.PI * radius;
  const ratio = capacity > 0 ? Math.min(active / capacity, 1) : 0;
  const dash = circumference * ratio;
  const over = percent !== null && percent > 100;

  return (
    <div className="relative flex items-center justify-center">
      <div aria-hidden className="gauge-halo" />
      <svg
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="-rotate-90"
        role="img"
        aria-label={`${active} vehículos dentro${capacity > 0 ? ` de ${capacity}` : ""}`}
      >
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={radius}
          fill="none"
          stroke="var(--color-elevated)"
          strokeWidth={STROKE}
        />
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={radius}
          fill="none"
          stroke={over ? "var(--color-danger)" : "var(--color-accent)"}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference - dash}`}
          className="gauge-progress"
        />
      </svg>

      <div className="absolute text-center">
        <p className="font-display text-4xl font-semibold tabular-nums text-gradient">
          {active}
        </p>
        <p className="text-xs text-muted">
          {capacity > 0 ? `de ${capacity} lugares` : "activos"}
        </p>
        {percent !== null && (
          <p
            className={`mt-1 text-xs font-semibold ${
              over ? "text-danger" : "text-accent"
            }`}
          >
            {percent}%
          </p>
        )}
      </div>
    </div>
  );
}
