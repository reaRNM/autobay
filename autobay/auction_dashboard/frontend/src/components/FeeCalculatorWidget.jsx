import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Divider,
  Grid,
  InputAdornment,
  Slider,
  Paper,
  CircularProgress
} from '@mui/material';
import { Calculate as CalculateIcon } from '@mui/icons-material';
import { useDashboard } from '../context/DashboardContext';

const FeeCalculatorWidget = () => {
  const { calculateFees } = useDashboard();
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [bidAmount, setBidAmount] = useState(100);
  const [buyerPremiumRate, setBuyerPremiumRate] = useState(15);
  const [salesTaxRate, setSalesTaxRate] = useState(7);
  const [shippingCost, setShippingCost] = useState(10);
  const [additionalFees, setAdditionalFees] = useState({
    handling: 0,
    insurance: 0
  });
  
  const [result, setResult] = useState(null);
  
  const handleCalculate = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await calculateFees({
        bid_amount: bidAmount,
        buyer_premium_rate: buyerPremiumRate / 100,
        sales_tax_rate: salesTaxRate / 100,
        shipping_cost: shippingCost,
        additional_fees: additionalFees
      });
      
      if (response.success) {
        setResult(response.data);
      } else {
        setError(response.error);
      }
    } catch (err) {
      setError('An error occurred during calculation');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleAdditionalFeeChange = (fee, value) => {
    setAdditionalFees(prev => ({
      ...prev,
      [fee]: parseFloat(value) || 0
    }));
  };
  
  return (
    <Box>
      <Grid container spacing={2}>
        {/* Bid Amount */}
        <Grid item xs={12}>
          <TextField
            label="Bid Amount"
            type="number"
            fullWidth
            value={bidAmount}
            onChange={(e) => setBidAmount(parseFloat(e.target.value) || 0)}
            InputProps={{
              startAdornment: <InputAdornment position="start">$</InputAdornment>,
            }}
          />
        </Grid>
        
        {/* Buyer Premium */}
        <Grid item xs={12}>
          <Typography gutterBottom>
            Buyer Premium: {buyerPremiumRate}%
          </Typography>
          <Slider
            value={buyerPremiumRate}
            onChange={(e, newValue) => setBuyerPremiumRate(newValue)}
            min={0}
            max={30}
            step={0.5}
            valueLabelDisplay="auto"
          />
        </Grid>
        
        {/* Sales Tax */}
        <Grid item xs={12}>
          <Typography gutterBottom>
            Sales Tax: {salesTaxRate}%
          </Typography>
          <Slider
            value={salesTaxRate}
            onChange={(e, newValue) => setSalesTaxRate(newValue)}
            min={0}
            max={15}
            step={0.1}
            valueLabelDisplay="auto"
          />
        </Grid>
        
        {/* Shipping Cost */}
        <Grid item xs={12}>
          <TextField
            label="Shipping Cost"
            type="number"
            fullWidth
            value={shippingCost}
            onChange={(e) => setShippingCost(parseFloat(e.target.value) || 0)}
            InputProps={{
              startAdornment: <InputAdornment position="start">$</InputAdornment>,
            }}
          />
        </Grid>
        
        {/* Additional Fees */}
        <Grid item xs={6}>
          <TextField
            label="Handling Fee"
            type="number"
            fullWidth
            value={additionalFees.handling}
            onChange={(e) => handleAdditionalFeeChange('handling', e.target.value)}
            InputProps={{
              startAdornment: <InputAdornment position="start">$</InputAdornment>,
            }}
          />
        </Grid>
        
        <Grid item xs={6}>
          <TextField
            label="Insurance"
            type="number"
            fullWidth
            value={additionalFees.insurance}
            onChange={(e) => handleAdditionalFeeChange('insurance', e.target.value)}
            InputProps={{
              startAdornment: <InputAdornment position="start">$</InputAdornment>,
            }}
          />
        </Grid>
        
        {/* Calculate Button */}
        <Grid item xs={12}>
          <Button
            variant="contained"
            color="primary"
            fullWidth
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <CalculateIcon />}
            onClick={handleCalculate}
            disabled={loading}
          >
            Calculate Total Cost
          </Button>
        </Grid>
      </Grid>
      
      {/* Error Message */}
      {error && (
        <Typography color="error" variant="body2" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}
      
      {/* Results */}
      {result && (
        <Box mt={3}>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Cost Breakdown
          </Typography>
          
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Grid container spacing={1}>
              <Grid item xs={8}>
                <Typography variant="body2">Bid Amount:</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="right">
                  ${result.bid_amount.toFixed(2)}
                </Typography>
              </Grid>
              
              <Grid item xs={8}>
                <Typography variant="body2">
                  Buyer Premium ({(result.buyer_premium_rate * 100).toFixed(1)}%):
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="right">
                  ${result.buyer_premium.toFixed(2)}
                </Typography>
              </Grid>
              
              <Grid item xs={8}>
                <Typography variant="body2">
                  Sales Tax ({(result.sales_tax_rate * 100).toFixed(1)}%):
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="right">
                  ${result.sales_tax.toFixed(2)}
                </Typography>
              </Grid>
              
              <Grid item xs={8}>
                <Typography variant="body2">Shipping Cost:</Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="right">
                  ${result.shipping_cost.toFixed(2)}
                </Typography>
              </Grid>
              
              {Object.entries(result.additional_fees).map(([key, value]) => (
                <React.Fragment key={key}>
                  <Grid item xs={8}>
                    <Typography variant="body2">
                      {key.charAt(0).toUpperCase() + key.slice(1)}:
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2" align="right">
                      ${value.toFixed(2)}
                    </Typography>
                  </Grid>
                </React.Fragment>
              ))}
              
              <Grid item xs={12}>
                <Divider sx={{ my: 1 }} />
              </Grid>
              
              <Grid item xs={8}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Total Cost:
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="subtitle1" fontWeight="bold" align="right">
                  ${result.total_cost.toFixed(2)}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default FeeCalculatorWidget;