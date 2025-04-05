T## roubleshooting

### Common Issues and Solutions

#### Database Connection Errors

**Issue**: Unable to connect to the PostgreSQL database.

**Solutions**:

- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check database credentials in `.env` file
- Ensure the database exists: `sudo -u postgres psql -c "\l"`
- Check PostgreSQL logs: `sudo tail -f /var/log/postgresql/postgresql-13-main.log`


#### API Server Won't Start

**Issue**: The API server fails to start.

**Solutions**:

- Check for port conflicts: `sudo lsof -i :8000`
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check log files in the `logs` directory
- Ensure environment variables are set correctly


#### Scraper Errors

**Issue**: Data collection tasks fail.

**Solutions**:

- Check internet connection
- Verify API keys are valid
- Look for rate limiting issues in logs
- Try running with fewer categories or pages: `python -m autobay.scrapers.hibid --max-pages=1`


#### Notification Issues

**Issue**: Not receiving notifications.

**Solutions**:

- Verify notification settings in `config/notifications.json`
- For Telegram, ensure the bot is added to your chat
- Check notification logs in the database
- Test notification service directly: `python -m autobay.utils.test_notifications`


### Debugging Tips

1. **Enable Debug Mode**:
Set `DEBUG=True` in your `.env` file for more detailed logs.
2. **Check Log Files**:

1. Application logs: `logs/autobay.log`
2. Error logs: `logs/error.log`
3. Workflow logs: Check the `workflow_logs` table in the database



3. **Database Inspection**:
Connect to the database to inspect tables:

```shellscript
psql -U autobay_user -d autobay
```

Useful commands:

1. `\dt` - List tables
2. `SELECT * FROM workflow_logs ORDER BY timestamp DESC LIMIT 10;` - View recent logs
3. `SELECT * FROM auction_items WHERE status = 'error';` - View items with errors



4. **Test Individual Components**:
Use the test scripts in the `tests` directory to test individual components:

```shellscript
python -m tests.test_profit_calculator
```