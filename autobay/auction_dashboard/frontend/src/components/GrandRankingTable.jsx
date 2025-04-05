import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Paper,
  Box,
  Chip,
  Typography,
  LinearProgress,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon
} from '@mui/icons-material';

// Helper function to get color based on risk score
const getRiskColor = (score) => {
  if (score < 0.3) return 'success';
  if (score < 0.6) return 'warning';
  return 'error';
};

// Helper function to get color based on profit margin
const getProfitColor = (margin) => {
  if (margin > 30) return 'success';
  if (margin > 15) return 'info';
  return 'warning';
};

const GrandRankingTable = ({ items = [] }) => {
  const [order, setOrder] = useState('desc');
  const [orderBy, setOrderBy] = useState('grand_ranking_score');

  const handleRequestSort = (property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const sortedItems = React.useMemo(() => {
    const comparator = (a, b) => {
      if (order === 'asc') {
        return a[orderBy] < b[orderBy] ? -1 : 1;
      } else {
        return a[orderBy] > b[orderBy] ? -1 : 1;
      }
    };
    return [...items].sort(comparator);
  }, [items, order, orderBy]);

  return (
    <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
      <Table stickyHeader size="small">
        <TableHead>
          <TableRow>
            <TableCell>
              <TableSortLabel
                active={orderBy === 'title'}
                direction={orderBy === 'title' ? order : 'asc'}
                onClick={() => handleRequestSort('title')}
              >
                Item
              </TableSortLabel>
            </TableCell>
            <TableCell align="right">
              <TableSortLabel
                active={orderBy === 'current_bid'}
                direction={orderBy === 'current_bid' ? order : 'asc'}
                onClick={() => handleRequestSort('current_bid')}
              >
                Current Bid
              </TableSortLabel>
            </TableCell>
            <TableCell align="right">
              <TableSortLabel
                active={orderBy === 'estimated_profit'}
                direction={orderBy === 'estimated_profit' ? order : 'asc'}
                onClick={() => handleRequestSort('estimated_profit')}
              >
                Est. Profit
              </TableSortLabel>
            </TableCell>
            <TableCell align="right">
              <TableSortLabel
                active={orderBy === 'profit_margin'}
                direction={orderBy === 'profit_margin' ? order : 'asc'}
                onClick={() => handleRequestSort('profit_margin')}
              >
                Margin
              </TableSortLabel>
            </TableCell>
            <TableCell align="right">
              <TableSortLabel
                active={orderBy === 'risk_score'}
                direction={orderBy === 'risk_score' ? order : 'asc'}
                onClick={() => handleRequestSort('risk_score')}
              >
                Risk
              </TableSortLabel>
            </TableCell>
            <TableCell align="right">
              <TableSortLabel
                active={orderBy === 'grand_ranking_score'}
                direction={orderBy === 'grand_ranking_score' ? order : 'asc'}
                onClick={() => handleRequestSort('grand_ranking_score')}
              >
                Score
              </TableSortLabel>
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedItems.map((item) => (
            <TableRow key={item.item_id} hover>
              <TableCell>
                <Box display="flex" alignItems="center">
                  <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                    {item.title}
                  </Typography>
                  <Tooltip title="View Details">
                    <IconButton size="small">
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Typography variant="caption" color="textSecondary">
                  {item.category}
                </Typography>
              </TableCell>
              <TableCell align="right">
                ${item.current_bid?.toFixed(2) || '0.00'}
              </TableCell>
              <TableCell align="right">
                <Box display="flex" alignItems="center" justifyContent="flex-end">
                  ${item.estimated_profit?.toFixed(2) || '0.00'}
                  {item.estimated_profit > 0 ? (
                    <TrendingUpIcon fontSize="small" color="success" sx={{ ml: 0.5 }} />
                  ) : (
                    <TrendingDownIcon fontSize="small" color="error" sx={{ ml: 0.5 }} />
                  )}
                </Box>
              </TableCell>
              <TableCell align="right">
                <Chip
                  label={`${item.profit_margin?.toFixed(1) || '0.0'}%`}
                  size="small"
                  color={getProfitColor(item.profit_margin)}
                />
              </TableCell>
              <TableCell align="right">
                <Tooltip title={`Risk Score: ${item.risk_score?.toFixed(2) || '0.00'}`}>
                  <Box sx={{ width: 60, ml: 'auto' }}>
                    <LinearProgress
                      variant="determinate"
                      value={item.risk_score * 100}
                      color={getRiskColor(item.risk_score)}
                      sx={{ height: 8, borderRadius: 5 }}
                    />
                  </Box>
                </Tooltip>
              </TableCell>
              <TableCell align="right">
                <Box sx={{ width: 60, ml: 'auto' }}>
                  <LinearProgress
                    variant="determinate"
                    value={item.grand_ranking_score * 100}
                    color="primary"
                    sx={{ height: 8, borderRadius: 5 }}
                  />
                  <Typography variant="caption" display="block" textAlign="center">
                    {(item.grand_ranking_score * 100).toFixed(0)}
                  </Typography>
                </Box>
              </TableCell>
            </TableRow>
          ))}
          {sortedItems.length === 0 && (
            <TableRow>
              <TableCell colSpan={6} align="center">
                No items found
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default GrandRankingTable;