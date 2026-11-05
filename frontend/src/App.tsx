// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./presentation/components/Layout";
import { RequireSession } from "./presentation/components/RequireSession";
import { Dashboard } from "./presentation/pages/Dashboard";
import { History } from "./presentation/pages/History";
import { Incidents } from "./presentation/pages/Incidents";
import { Stats } from "./presentation/pages/Stats";
import { Cameras } from "./presentation/pages/Cameras";
import { Model } from "./presentation/pages/Model";
import { Users } from "./presentation/pages/Users";
import { VehicleDetail } from "./presentation/pages/VehicleDetail";
import { Whitelist } from "./presentation/pages/Whitelist";
import { RequireRole } from "./presentation/components/RequireRole";

export default function App() {
  return (
    <Routes>
      <Route
        element={
          <RequireSession>
            <Layout />
          </RequireSession>
        }
      >
        <Route path="/" element={<Dashboard />} />
        <Route path="/stats" element={<Stats />} />
        <Route path="/incidents" element={<Incidents />} />
        <Route path="/vehicles/:id" element={<VehicleDetail />} />
        <Route path="/history" element={<History />} />
        <Route
          path="/whitelist"
          element={
            <RequireRole roles={["admin", "operator"]}>
              <Whitelist />
            </RequireRole>
          }
        />
        <Route
          path="/model"
          element={
            <RequireRole roles={["admin"]}>
              <Model />
            </RequireRole>
          }
        />
        <Route
          path="/cameras"
          element={
            <RequireRole roles={["admin"]}>
              <Cameras />
            </RequireRole>
          }
        />
        <Route
          path="/users"
          element={
            <RequireRole roles={["admin"]}>
              <Users />
            </RequireRole>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
