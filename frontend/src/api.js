import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchRiskScores = (method = 'gat', page = 1, pageSize = 50) =>
  api.get(`/risk-scores`, { params: { method, page, page_size: pageSize } });

export const fetchGraph = (topN = 50) =>
  api.get(`/graph`, { params: { top_n: topN } });

export const fetchExplain = (workIndex) =>
  api.get(`/explain/${workIndex}`);

export const postLabel = (workIndex, label) =>
  api.post(`/label`, { work_index: workIndex, label });

export default api;
