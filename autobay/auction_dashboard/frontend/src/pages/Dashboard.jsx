import React, { useState } from 'react';
import { 
  Container, 
  Grid, 
  Paper, 
  Typography, 
  Box, 
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Divider
} from '@mui/material';
import { 
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  AttachMoney as MoneyIcon,
  Assessment as AssessmentIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import { useDashboard } from '../context/DashboardContext';

// Components
import GrandRankingTable from '../components/GrandRankingTable';
import ProfitChart from '../components/ProfitChart';
import AuctionMetricsCard from '../components/AuctionMetricsCard';
import RiskMetricsCard from '../components/RiskMetricsCard';
import FeeCalculatorWidget from '../components/FeeCalculatorWidget';
import RecentHistoryTable from '../components/RecentHistoryTable';

const Dashboard = () => {
  const { 
    dashboardData, 
    timeRange, 
    setTimeRange, 
    loading, 
    error, 
    refreshDashboard 
  } = useDashboard();

  const handleTimeRangeChange = (event) => {
    setTimeRange(event.target.value);
  };

  if (loading && !dashboardData) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        minHeight="80vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        minHeight="80vh"
      >
        <Typography color="error" variant="h6">
          Error loading dashboard data: {error}
        </Typography>
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Auction Dashboard
        </Typography>
        
        <Box display="flex" alignItems="center">
          <FormControl variant="outlined" size="small" sx={{ minWidth: 120, mr: 2 }}>
            <InputLabel id="time-range-label">Time Range</InputLabel>
            <Select
              labelId="time-range-label"
              id="time-range"
              value={timeRange}
              onChange={handleTimeRangeChange}
              label="Time Range"
            >
              <MenuItem value="day">Last 24 Hours</MenuItem>
              <MenuItem value="week">Last Week</MenuItem>
              <MenuItem value="month">Last Month</MenuItem>
              <MenuItem value="year">Last Year</MenuItem>
              <MenuItem value="all">All Time</MenuItem>
            </Select>
          </FormControl>
          
          <Button 
            variant="contained" 
            startIcon={<RefreshIcon />}
            onClick={refreshDashboard}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Metrics Overview */}
      <Grid container spacing={3} mb={4}>
        {/* Profit Metrics */}
        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: 'primary.light',
              color: 'white',
            }}
          >
            <Box display="flex" alignItems="center" mb={1}>
              <MoneyIcon sx={{ mr: 1 }} />
              <Typography variant="h6" component="div">
                Total Profit
              </Typography>
            </Box>
            <Typography variant="h3" component="div" sx={{ fontWeight: 'bold', mt: 1 }}>
              ${dashboardData?.profit_metrics?.total_profit?.toFixed(2) || '0.00'}
            </Typography>
            <Typography variant="body2" sx={{ mt: 'auto' }}>
              {dashboardData?.profit_metrics?.total_sold || 0} items sold
            </Typography>
          </Paper>
        </Grid>

        {/* ROI Metrics */}
        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: 'success.light',
              color: 'white',
            }}
          >
            <Box display="flex" alignItems="center" mb={1}>
              <TrendingUpIcon sx={{ mr: 1 }} />
              <Typography variant="h6" component="div">
                Average ROI
              </Typography>
            </Box>
            <Typography variant="h3" component="div" sx={{ fontWeight: 'bold', mt: 1 }}>
              {dashboardData?.profit_metrics?.average_roi?.toFixed(2) || '0.00'}%
            </Typography>
            <Typography variant="body2" sx={{ mt: 'auto' }}>
              Avg. Profit: ${dashboardData?.profit_metrics?.average_profit?.toFixed(2) || '0.00'}
            </Typography>
          </Paper>
        </Grid>

        {/* Auction Metrics */}
        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: 'info.light',
              color: 'white',
            }}
          >
            <Box display="flex" alignItems="center" mb={1}>
              <AssessmentIcon sx={{ mr: 1 }} />
              <Typography variant="h6" component="div">
                Auctions Won
              </Typography>
            </Box>
            <Typography variant="h3" component="div" sx={{ fontWeight: 'bold', mt: 1 }}>
              {dashboardData?.auction_metrics?.total_won || 0}
            </Typography>
            <Typography variant="body2" sx={{ mt: 'auto' }}>
              Total Value: ${dashboardData?.auction_metrics?.total_purchase_value?.toFixed(2) || '0.00'}
            </Typography>
          </Paper>
        </Grid>

        {/* Risk Metrics */}
        <Grid item xs={12} md={3}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 140,
              bgcolor: 'warning.light',
              color: 'white',
            }}
          >
            <Box display="flex" alignItems="center" mb={1}>
              <WarningIcon sx={{ mr: 1 }} />
              <Typography variant="h6" component="div">
                Avg. Risk Score
              </Typography>
            </Box>
            <Typography variant="h3" component="div" sx={{ fontWeight: 'bold', mt: 1 }}>
              {dashboardData?.risk_metrics?.average_risk_score?.toFixed(2) || '0.00'}
            </Typography>
            <Typography variant="body2" sx={{ mt: 'auto' }}>
              Based on {dashboardData?.risk_metrics?.risk_distribution?.length || 0} items
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Left Column */}
        <Grid item xs={12} md={8}>
          {/* Profit Chart */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" component="h2" gutterBottom>
              Profit Trends
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <ProfitChart data={dashboardData?.profit_metrics?.profit_over_time || []} />
          </Paper>

          {/* Grand Ranking Table */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" component="h2" gutterBottom>
              Grand Ranking
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <GrandRankingTable items={dashboardData?.top_ranked_items || []} />
          </Paper>

          {/* Recent History */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" component="h2" gutterBottom>
              Recent Auction History
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <RecentHistoryTable history={dashboardData?.recent_history || []} />
          </Paper>
        </Grid>

        {/* Right Column */}
        <Grid item xs={12} md={4}>
          {/* Fee Calculator Widget */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" component="h2" gutterBottom>
              Fee Calculator
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <FeeCalculatorWidget />
          </Paper>

          {/* Auction Metrics */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" component="h2" gutterBottom>
              Auction Metrics
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <AuctionMetricsCard metrics={dashboardData?.auction_metrics} />
          </Paper>

          {/* Risk Metrics */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" component="h2" gutterBottom>
              Risk Analysis
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <RiskMetricsCard metrics={dashboardData?.risk_metrics} />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;