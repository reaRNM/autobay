import React, { createContext, useState, useEffect, useContext } from 'react';
import api from '../services/api';
import { useAuth } from './AuthContext';

const DashboardContext = createContext();

export const useDashboard = () => useContext(DashboardContext);

export const DashboardProvider = ({ children }) => {
  const { isAuthenticated } = useAuth();
  const [dashboardData, setDashboardData] = useState(null);
  const [timeRange, setTimeRange] = useState('month');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Fetch dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      if (!isAuthenticated) return;

      setLoading(true);
      setError(null);
      try {
        const response = await api.get(`/api/dashboard?time_range=${timeRange}`);
        setDashboardData(response.data);
      } catch (err) {
        console.error('Failed to fetch dashboard data:', err);
        setError(err.response?.data?.message || 'Failed to fetch dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [isAuthenticated, timeRange, refreshTrigger]);

  // Refresh dashboard data
  const refreshDashboard = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  // Calculate fees
  const calculateFees = async (data) => {
    try {
      const response = await api.post('/api/fees/calculate', data);
      return { success: true, data: response.data };
    } catch (err) {
      console.error('Fee calculation failed:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Fee calculation failed' 
      };
    }
  };

  // Calculate profit
  const calculateProfit = async (data) => {
    try {
      const response = await api.post('/api/profit/calculate', data);
      return { success: true, data: response.data };
    } catch (err) {
      console.error('Profit calculation failed:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Profit calculation failed' 
      };
    }
  };

  // Get auction history
  const getAuctionHistory = async (params = {}) => {
    try {
      const response = await api.get('/api/history', { params });
      return { success: true, data: response.data };
    } catch (err) {
      console.error('Failed to fetch auction history:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Failed to fetch auction history' 
      };
    }
  };

  // Get alert configurations
  const getAlertConfigs = async () => {
    try {
      const response = await api.get('/api/alerts/config');
      return { success: true, data: response.data };
    } catch (err) {
      console.error('Failed to fetch alert configurations:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Failed to fetch alert configurations' 
      };
    }
  };

  // Create alert configuration
  const createAlertConfig = async (data) => {
    try {
      const response = await api.post('/api/alerts/config', data);
      return { success: true, data: response.data };
    } catch (err) {
      console.error('Failed to create alert configuration:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Failed to create alert configuration' 
      };
    }
  };

  // Update alert configuration
  const updateAlertConfig = async (id, data) => {
    try {
      const response = await api.put(`/api/alerts/config/${id}`, data);
      return { success: true, data: response.data };
    } catch (err) {
      console.error('Failed to update alert configuration:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Failed to update alert configuration' 
      };
    }
  };

  // Delete alert configuration
  const deleteAlertConfig = async (id) => {
    try {
      await api.delete(`/api/alerts/config/${id}`);
      return { success: true };
    } catch (err) {
      console.error('Failed to delete alert configuration:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Failed to delete alert configuration' 
      };
    }
  };

  // Test alert
  const testAlert = async (data) => {
    try {
      const response = await api.post('/api/alerts/test', data);
      return { success: true, data: response.data };
    } catch (err) {
      console.error('Failed to test alert:', err);
      return { 
        success: false, 
        error: err.response?.data?.message || 'Failed to test alert' 
      };
    }
  };

  const value = {
    dashboardData,
    timeRange,
    setTimeRange,
    loading,
    error,
    refreshDashboard,
    calculateFees,
    calculateProfit,
    getAuctionHistory,
    getAlertConfigs,
    createAlertConfig,
    updateAlertConfig,
    deleteAlertConfig,
    testAlert,
  };

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
};