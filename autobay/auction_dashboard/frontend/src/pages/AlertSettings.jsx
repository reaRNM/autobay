import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Button,
  Grid,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Switch,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  FormControlLabel,
  Checkbox,
  InputAdornment
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Notifications as NotificationsIcon,
  Send as SendIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { useDashboard } from '../context/DashboardContext';

const AlertSettings = () => {
  const { getAlertConfigs, createAlertConfig, updateAlertConfig, deleteAlertConfig, testAlert } = useDashboard();
  const { enqueueSnackbar } = useSnackbar();
  
  const [loading, setLoading] = useState(true);
  const [alertConfigs, setAlertConfigs] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    alert_type: 'auction_opportunity',
    conditions: {},
    notification_channels: ['telegram'],
    is_active: true
  });
  
  const [testData, setTestData] = useState({
    alert_type: 'test_alert',
    message: 'This is a test alert',
    channels: ['telegram'],
    data: {
      item: {
        title: 'Test Item',
        current_bid: 100,
        estimated_profit: 30,
        risk_score: 0.3
      }
    }
  });
  
  // Fetch alert configurations
  useEffect(() => {
    const fetchAlertConfigs = async () => {
      setLoading(true);
      try {
        const response = await getAlertConfigs();
        if (response.success) {
          setAlertConfigs(response.data);
        } else {
          enqueueSnackbar(`Error: ${response.error}`, { variant: 'error' });
        }
      } catch (error) {
        console.error('Failed to fetch alert configurations:', error);
        enqueueSnackbar('Failed to fetch alert configurations', { variant: 'error' });
      } finally {
        setLoading(false);
      }
    };
    
    fetchAlertConfigs();
  }, [getAlertConfigs, enqueueSnackbar]);
  
  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
  
  // Handle checkbox changes
  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setFormData(prev => ({ ...prev, [name]: checked }));
  };
  
  // Handle notification channel changes
  const handleChannelChange = (e) => {
    setFormData(prev => ({ ...prev, notification_channels: e.target.value }));
  };
  
  // Handle condition changes
  const handleConditionChange = (key, value, operator = '>=') => {
    setFormData(prev => ({
      ...prev,
      conditions: {
        ...prev.conditions,
        [key]: { operator, value: parseFloat(value) || 0 }
      }
    }));
  };
  
  // Open dialog for creating a new alert config
  const handleOpenCreateDialog = () => {
    setEditingConfig(null);
    setFormData({
      name: '',
      description: '',
      alert_type: 'auction_opportunity',
      conditions: {
        estimated_profit: { operator: '>=', value: 20 },
        risk_score: { operator: '<=', value: 0.5 }
      },
      notification_channels: ['telegram'],
      is_active: true
    });
    setDialogOpen(true);
  };
  
  // Open dialog for editing an existing alert config
  const handleOpenEditDialog = (config) => {
    setEditingConfig(config);
    setFormData({
      name: config.name,
      description: config.description || '',
      alert_type: config.alert_type,
      conditions: config.conditions,
      notification_channels: config.notification_channels,
      is_active: config.is_active
    });
    setDialogOpen(true);
  };
  
  // Close the dialog
  const handleCloseDialog = () => {
    setDialogOpen(false);
  };
  
  // Save the alert configuration
  const handleSaveConfig = async () => {
    try {
      let response;
      
      if (editingConfig) {
        // Update existing config
        response = await updateAlertConfig(editingConfig.id, formData);
      } else {
        // Create new config
        response = await createAlertConfig(formData);
      }
      
      if (response.success) {
        enqueueSnackbar(
          `Alert configuration ${editingConfig ? 'updated' : 'created'} successfully`,
          { variant: 'success' }
        );
        
        // Refresh the list
        const configsResponse = await getAlertConfigs();
        if (configsResponse.success) {
          setAlertConfigs(configsResponse.data);
        }
        
        // Close the dialog
        setDialogOpen(false);
      } else {
        enqueueSnackbar(`Error: ${response.error}`, { variant: 'error' });
      }
    } catch (error) {
      console.error('Failed to save alert configuration:', error);
      enqueueSnackbar('Failed to save alert configuration', { variant: 'error' });
    }
  };
  
  // Delete an alert configuration
  const handleDeleteConfig = async (id) => {
    if (window.confirm('Are you sure you want to delete this alert configuration?')) {
      try {
        const response = await deleteAlertConfig(id);
        
        if (response.success) {
          enqueueSnackbar('Alert configuration deleted successfully', { variant: 'success' });
          
          // Remove from the list
          setAlertConfigs(prev => prev.filter(config => config.id !== id));
        } else {
          enqueueSnackbar(`Error: ${response.error}`, { variant: 'error' });
        }
      } catch (error) {
        console.error('Failed to delete alert configuration:', error);
        enqueueSnackbar('Failed to delete alert configuration', { variant: 'error' });
      }
    }
  };
  
  // Toggle alert configuration active state
  const handleToggleActive = async (config) => {
    try {
      const response = await updateAlertConfig(config.id, {
        ...config,
        is_active: !config.is_active
      });
      
      if (response.success) {
        enqueueSnackbar(
          `Alert ${response.data.is_active ? 'activated' : 'deactivated'} successfully`,
          { variant: 'success' }
        );
        
        // Update the list
        setAlertConfigs(prev => prev.map(item => 
          item.id === config.id ? { ...item, is_active: !item.is_active } : item
        ));
      } else {
        enqueueSnackbar(`Error: ${response.error}`, { variant: 'error' });
      }
    } catch (error) {
      console.error('Failed to toggle alert state:', error);
      enqueueSnackbar('Failed to toggle alert state', { variant: 'error' });
    }
  };
  
  // Open test alert dialog
  const handleOpenTestDialog = () => {
    setTestDialogOpen(true);
  };
  
  // Close test alert dialog
  const handleCloseTestDialog = () => {
    setTestDialogOpen(false);
  };
  
  // Handle test data changes
  const handleTestDataChange = (e) => {
    const { name, value } = e.target;
    setTestData(prev => ({ ...prev, [name]: value }));
  };
  
  // Handle test channel changes
  const handleTestChannelChange = (e) => {
    setTestData(prev => ({ ...prev, channels: e.target.value }));
  };
  
  // Send test alert
  const handleSendTestAlert = async () => {
    try {
      const response = await testAlert(testData);
      
      if (response.success) {
        enqueueSnackbar('Test alert sent successfully', { variant: 'success' });
        setTestDialogOpen(false);
      } else {
        enqueueSnackbar(`Error: ${response.error}`, { variant: 'error' });
      }
    } catch (error) {
      console.error('Failed to send test alert:', error);
      enqueueSnackbar('Failed to send test alert', { variant: 'error' });
    }
  };
  
  // Render alert type label
  const renderAlertTypeLabel = (type) => {
    switch (type) {
      case 'auction_opportunity':
        return 'Auction Opportunity';
      case 'bid_spike':
        return 'Bid Spike';
      case 'auction_ending':
        return 'Auction Ending';
      case 'price_drop':
        return 'Price Drop';
      default:
        return type;
    }
  };
  
  // Render notification channels
  const renderNotificationChannels = (channels) => {
    return channels.map(channel => (
      <Chip 
        key={channel} 
        label={channel.charAt(0).toUpperCase() + channel.slice(1)} 
        size="small" 
        color="primary" 
        variant="outlined" 
        sx={{ mr: 0.5 }} 
      />
    ));
  };
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Alert Settings
        </Typography>
        
        <Box>
          <Button 
            variant="outlined" 
            startIcon={<SendIcon />} 
            onClick={handleOpenTestDialog}
            sx={{ mr: 2 }}
          >
            Test Alert
          </Button>
          <Button 
            variant="contained" 
            startIcon={<AddIcon />} 
            onClick={handleOpenCreateDialog}
          >
            New Alert
          </Button>
        </Box>
      </Box>
      
      {/* Main Content */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" component="h2" gutterBottom>
          <NotificationsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Alert Configurations
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        {loading ? (
          <Box display="flex" justifyContent="center" p={3}>
            <CircularProgress />
          </Box>
        ) : alertConfigs.length === 0 ? (
          <Box textAlign="center" p={3}>
            <Typography variant="body1" color="textSecondary">
              No alert configurations found. Create your first alert to get notified about auction opportunities.
            </Typography>
          </Box>
        ) : (
          <List>
            {alertConfigs.map((config) => (
              <React.Fragment key={config.id}>
                <ListItem>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center">
                        {config.name}
                        {!config.is_active && (
                          <Chip 
                            label="Inactive" 
                            size="small" 
                            color="default" 
                            sx={{ ml: 1 }} 
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography variant="body2" color="textSecondary">
                          {renderAlertTypeLabel(config.alert_type)}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          {config.description || 'No description'}
                        </Typography>
                        <Box mt={1}>
                          {renderNotificationChannels(config.notification_channels)}
                        </Box>
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={config.is_active}
                          onChange={() => handleToggleActive(config)}
                          color="primary"
                        />
                      }
                      label=""
                    />
                    <IconButton 
                      edge="end" 
                      aria-label="edit"
                      onClick={() => handleOpenEditDialog(config)}
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton 
                      edge="end" 
                      aria-label="delete"
                      onClick={() => handleDeleteConfig(config.id)}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
              </React.Fragment>
            ))}
          </List>
        )}
      </Paper>
      
      {/* Alert Configuration Dialog */}
      <Dialog open={dialogOpen} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingConfig ? 'Edit Alert Configuration' : 'Create Alert Configuration'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                name="name"
                label="Alert Name"
                fullWidth
                value={formData.name}
                onChange={handleInputChange}
                required
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                name="description"
                label="Description"
                fullWidth
                multiline
                rows={2}
                value={formData.description}
                onChange={handleInputChange}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Alert Type</InputLabel>
                <Select
                  name="alert_type"
                  value={formData.alert_type}
                  onChange={handleInputChange}
                  label="Alert Type"
                >
                  <MenuItem value="auction_opportunity">Auction Opportunity</MenuItem>
                  <MenuItem value="bid_spike">Bid Spike</MenuItem>
                  <MenuItem value="auction_ending">Auction Ending</MenuItem>
                  <MenuItem value="price_drop">Price Drop</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Notification Channels</InputLabel>
                <Select
                  multiple
                  name="notification_channels"
                  value={formData.notification_channels}
                  onChange={handleChannelChange}
                  label="Notification Channels"
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} />
                      ))}
                    </Box>
                  )}
                >
                  <MenuItem value="telegram">Telegram</MenuItem>
                  <MenuItem value="email">Email</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Alert Conditions
              </Typography>
              <Divider sx={{ mb: 2 }} />
            </Grid>
            
            {formData.alert_type === 'auction_opportunity' && (
              <>
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Minimum Estimated Profit"
                    type="number"
                    fullWidth
                    value={formData.conditions.estimated_profit?.value || ''}
                    onChange={(e) => handleConditionChange('estimated_profit', e.target.value, '>=')}
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                    }}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    label="Maximum Risk Score"
                    type="number"
                    fullWidth
                    value={formData.conditions.risk_score?.value || ''}
                    onChange={(e) => handleConditionChange('risk_score', e.target.value, '<=')}
                    inputProps={{ min: 0, max: 1, step: 0.1 }}
                  />
                </Grid>
              </>
            )}
            
            {formData.alert_type === 'bid_spike' && (
              <Grid item xs={12} md={6}>
                <TextField
                  label="Minimum Bid Increase Percentage"
                  type="number"
                  fullWidth
                  value={formData.conditions.bid_increase_percentage?.value || ''}
                  onChange={(e) => handleConditionChange('bid_increase_percentage', e.target.value, '>=')}
                  InputProps={{
                    endAdornment: <InputAdornment position="end">%</InputAdornment>,
                  }}
                />
              </Grid>
            )}
            
            {formData.alert_type === 'auction_ending' && (
              <Grid item xs={12} md={6}>
                <TextField
                  label="Minutes Before End"
                  type="number"
                  fullWidth
                  value={formData.conditions.minutes_remaining?.value || ''}
                  onChange={(e) => handleConditionChange('minutes_remaining', e.target.value, '<=')}
                />
              </Grid>
            )}
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={formData.is_active}
                    onChange={handleCheckboxChange}
                    name="is_active"
                    color="primary"
                  />
                }
                label="Active"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSaveConfig} variant="contained" color="primary">
            Save
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Test Alert Dialog */}
      <Dialog open={testDialogOpen} onClose={handleCloseTestDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Send Test Alert</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                name="message"
                label="Alert Message"
                fullWidth
                value={testData.message}
                onChange={handleTestDataChange}
                required
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Alert Type</InputLabel>
                <Select
                  name="alert_type"
                  value={testData.alert_type}
                  onChange={handleTestDataChange}
                  label="Alert Type"
                >
                  <MenuItem value="test_alert">Test Alert</MenuItem>
                  <MenuItem value="auction_opportunity">Auction Opportunity</MenuItem>
                  <MenuItem value="bid_spike">Bid Spike</MenuItem>
                  <MenuItem value="auction_ending">Auction Ending</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Notification Channels</InputLabel>
                <Select
                  multiple
                  name="channels"
                  value={testData.channels}
                  onChange={handleTestChannelChange}
                  label="Notification Channels"
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} />
                      ))}
                    </Box>
                  )}
                >
                  <MenuItem value="telegram">Telegram</MenuItem>
                  <MenuItem value="email">Email</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseTestDialog}>Cancel</Button>
          <Button onClick={handleSendTestAlert} variant="contained" color="primary">
            Send Test Alert
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default AlertSettings;