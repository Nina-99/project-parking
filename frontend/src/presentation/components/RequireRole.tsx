// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ShieldAlert } from "lucide-react";
import { useSession } from "../../application/session";
import { ROLE_LABELS, type UserRole } from "../../domain/types";

function Forbidden({ role, roles }: { role: UserRole; roles: UserRole[] }) {
  const allowed = roles.map((item) => ROLE_LABELS[item]).join(" o ");
  return (
    <div className="card mx-auto max-w-md text-center">
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-warn/10 text-warn">
        <ShieldAlert className="h-6 w-6" />
      </div>
      <h2 className="mt-4 font-display text-xl font-semibold text-ink">
        Sin permiso
      </h2>
      <p className="mt-2 text-sm text-muted">
        Tu rol es {ROLE_LABELS[role]} y esta sección requiere {allowed}.
      </p>
      <Link to="/" className="btn-secondary mt-5">
        Volver al dashboard
      </Link>
    </div>
  );
}

export function RequireRole({
  roles,
  children,
}: {
  roles: UserRole[];
  children: ReactNode;
}) {
  const { user } = useSession();
  if (!user) return null;
  if (!roles.includes(user.role)) return <Forbidden role={user.role} roles={roles} />;
  return <>{children}</>;
}
