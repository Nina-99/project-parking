// SPDX-FileCopyrightText: 2026 Nina
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
        <Icon className="w-5 h-5 text-subtle" />
      </div>
      <span className="stat-value">{value}</span>
      {sub && <span className="text-xs text-subtle">{sub}</span>}
    </div>
  );
}
