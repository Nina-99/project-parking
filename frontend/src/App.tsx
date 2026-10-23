import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./presentation/components/Layout";
import { Dashboard } from "./presentation/pages/Dashboard";
import { History } from "./presentation/pages/History";
import { Incidents } from "./presentation/pages/Incidents";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/incidents" element={<Incidents />} />
        <Route path="/history" element={<History />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
