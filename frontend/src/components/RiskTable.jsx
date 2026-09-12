import React, { useState, useEffect } from 'react';
import { fetchRiskScores } from '../api';
import { ChevronLeft, ChevronRight, AlertCircle, CheckCircle, HelpCircle } from 'lucide-react';
import './RiskTable.css';

const BADGE_COLORS = {
  'TRAINING LABEL': 'bg-blue-100 text-blue-800 border-blue-200',
  'GAT-TOP': 'bg-purple-100 text-purple-800 border-purple-200',
  'NEW': 'bg-gray-100 text-gray-800 border-gray-200',
};

const RiskTable = ({ onRowClick }) => {
  const [data, setData] = useState([]);
  const [method, setMethod] = useState('gat');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const response = await fetchRiskScores(method, page);
        setData(response.data.results);
        setTotalPages(response.data.total_pages);
      } catch (error) {
        console.error('Error fetching risk scores:', error);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [method, page]);

  return (
    <div className="risk-table-container">
      <div className="table-controls">
        <div className="method-toggle">
          <button
            className={method === 'gat' ? 'active' : ''}
            onClick={() => { setMethod('gat'); setPage(1); }}
          >
            GAT
          </button>
          <button
            className={method === 'siamese' ? 'active' : ''}
            onClick={() => { setMethod('siamese'); setPage(1); }}
          >
            Siamese
          </button>
        </div>

        <div className="pagination">
          <button
            disabled={page === 1}
            onClick={() => setPage(p => p - 1)}
          >
            <ChevronLeft size={16} /> Prev
          </button>
          <span>Page {page} of {totalPages}</span>
          <button
            disabled={page === totalPages}
            onClick={() => setPage(p => p + 1)}
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading risk scores...</div>
      ) : (
        <div className="table-wrapper">
          <table className="risk-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>MP</th>
                <th>IDA</th>
                <th>Amount</th>
                <th>Lag</th>
                <th>Status</th>
                <th>Score</th>
                <th>Label Status</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr
                  key={row.work_index}
                  onClick={() => onRowClick(row)}
                  className="clickable-row"
                >
                  <td>{(page - 1) * 50 + index + 1}</td>
                  <td>{row.mp_name}</td>
                  <td>{row.ida}</td>
                  <td>₹{row.amount.toLocaleString()}</td>
                  <td>{row.lag} days</td>
                  <td>{row.status}</td>
                  <td className="score-cell">{row.score.toFixed(4)}</td>
                  <td>
                    <span className={`badge ${BADGE_COLORS[row.label_status] || BADGE_COLORS.NEW}`}>
                      {row.label_status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default RiskTable;
