// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import type { ComponentType } from "react";

export function StatCard({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="stat-card">
      <div className="flex items-center justify-between">
        <span className="stat-label">{label}</span>
        <span className="icon-chip h-9 w-9">
          <Icon className="h-4 w-4" />
        </span>
      </div>
      <span className="stat-value text-gradient">{value}</span>
      {sub && <span className="text-xs text-subtle">{sub}</span>}
    </div>
  );
}
