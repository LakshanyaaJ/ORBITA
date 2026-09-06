import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AuthPage from './pages/AuthPage';
import PhoneWebcam from './pages/PhoneWebcam';
import MissionHome from './pages/MissionHome';
import ExperimentBrowser from './pages/ExperimentBrowser';
import ExperimentBriefing from './pages/ExperimentBriefing';
import LiveExperiment from './pages/LiveExperiment';
import ExperimentLog from './pages/ExperimentLog';
import SystemDiagnostics from './pages/SystemDiagnostics';
import PageShell from './components/common/PageShell';
import MissionControlPage from './mission-control/pages/Home';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/auth" element={<AuthPage />} />
        <Route path="/cam" element={<PhoneWebcam />} />
        <Route path="/phone-cam" element={<Navigate to="/cam" replace />} />
        
        {/* Protected Operational Routes */}
        <Route element={<PageShell />}>
          <Route path="/mission" element={<MissionHome />} />
          <Route path="/experiments" element={<ExperimentBrowser />} />
          <Route path="/experiments/:id/briefing" element={<ExperimentBriefing />} />
          <Route path="/experiments/:id/live" element={<LiveExperiment />} />
          <Route path="/experiments/:id/log" element={<ExperimentLog />} />
          <Route path="/system" element={<SystemDiagnostics />} />
          <Route path="/dataset" element={<SystemDiagnostics />} />
          <Route path="/vdata" element={<SystemDiagnostics />} />
        </Route>
        
        {/* Unified Mission Control Route */}
        <Route path="/mission-control" element={<MissionControlPage />} />
        <Route path="/orbita-mission-control" element={<Navigate to="/mission-control" replace />} />
        
        <Route path="*" element={<Navigate to="/auth" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
