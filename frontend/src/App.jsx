import React, { useState } from 'react';
import RiskTable from './components/RiskTable';
import './App.css';

function App() {
  const [selectedRow, setSelectedRow] = useState(null);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>MPLADS Fraud Detection Dashboard</h1>
      </header>
      <main>
        <RiskTable onRowClick={setSelectedRow} />
        <div className="details-panel">
          {selectedRow ? (
            <div>Selected: {selectedRow.work_index}</div>
          ) : (
            <div className="placeholder">Select a row to see details</div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
