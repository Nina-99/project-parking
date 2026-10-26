// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./presentation/components/Layout";
import { RequireSession } from "./presentation/components/RequireSession";
import { Dashboard } from "./presentation/pages/Dashboard";
import { History } from "./presentation/pages/History";
import { Incidents } from "./presentation/pages/Incidents";
import { Model } from "./presentation/pages/Model";
import { VehicleDetail } from "./presentation/pages/VehicleDetail";

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
        <Route path="/incidents" element={<Incidents />} />
        <Route path="/vehicles/:id" element={<VehicleDetail />} />
        <Route path="/history" element={<History />} />
        <Route path="/model" element={<Model />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
