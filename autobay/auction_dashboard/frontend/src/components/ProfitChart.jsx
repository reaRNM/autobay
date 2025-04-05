import React from 'react';
import { Box, useTheme } from '@mui/material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Bar,
  BarChart,
  ComposedChart
} from 'recharts';

const ProfitChart = ({ data = [] }) => {
  const theme = useTheme();
  
  // Format data for the chart
  const formattedData = React.useMemo(() => {
    return data.map(item => ({
      ...item,
      period: item.period,
      profit: parseFloat(item.profit) || 0,
      count: parseInt(item.count) || 0
    }));
  }, [data]);
  
  if (formattedData.length === 0) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        height={300}
        bgcolor="grey.100"
        borderRadius={1}
      >
        No profit data available
      </Box>
    );
  }
  
  return (
    <Box height={300}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={formattedData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="period" />
          <YAxis yAxisId="left" orientation="left" stroke={theme.palette.primary.main} />
          <YAxis yAxisId="right" orientation="right" stroke={theme.palette.secondary.main} />
          <Tooltip 
            formatter={(value, name) => {
              if (name === 'profit') return [`$${value.toFixed(2)}`, 'Profit'];
              if (name === 'count') return [value, 'Items Sold'];
              return [value, name];
            }}
          />
          <Legend />
          <Bar 
            yAxisId="right" 
            dataKey="count" 
            fill={theme.palette.secondary.light} 
            name="Items Sold" 
            barSize={20}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="profit"
            stroke={theme.palette.primary.main}
            activeDot={{ r: 8 }}
            name="Profit"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ProfitChart;