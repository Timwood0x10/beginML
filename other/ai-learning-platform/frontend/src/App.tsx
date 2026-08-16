import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import BrowsePage from "./pages/BrowsePage";
import MapPage from "./pages/MapPage";
import PathPage from "./pages/PathPage";
import NotePage from "./pages/NotePage";
import LabPage from "./lab/LabPage";
import AgentPage from "./pages/AgentPage";
import PaperPage from "./pages/PaperPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="/browse" element={<BrowsePage />} />
        <Route path="/map" element={<MapPage />} />
        <Route path="/path" element={<PathPage />} />
        <Route path="/agent" element={<AgentPage />} />
        <Route path="/paper" element={<PaperPage />} />
        <Route path="/lab" element={<LabPage />} />
        <Route path="/lab/:moduleId" element={<LabPage />} />
        <Route path="/note/:id" element={<NotePage />} />
        <Route path="*" element={<HomePage />} />
      </Route>
    </Routes>
  );
}
