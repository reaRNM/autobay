## API Documentation

### Base URL

`http://localhost:8000/api/v1`

### Authentication

Most endpoints require authentication using JWT tokens.

**Login**:

- **URL**: `/auth/login`
- **Method**: `POST`
- **Body**:

```json
{
  "username": "your_username",
  "password": "your_password"
}
```


- **Response**:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```




**For authenticated requests, include the token in the Authorization header**:

```plaintext
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Endpoints

#### Profit Calculator

**Calculate Profit**:

- **URL**: `/profit/calculate`
- **Method**: `POST`
- **Body**:

```json
{
  "item_url": "https://www.ebay.com/itm/123456789",
  "estimated_buy_price": 50.00,
  "condition": "used_good",
  "category": "electronics"
}
```


- **Response**:

```json
{
  "item_id": "550e8400-e29b-41d4-a716-446655440000",
  "estimated_buy_price": 50.00,
  "estimated_sell_price": 89.99,
  "estimated_fees": 12.55,
  "estimated_shipping": 8.75,
  "estimated_profit": 18.69,
  "estimated_roi": 0.37,
  "confidence_score": 0.85
}
```




#### Bid Intelligence

**Get Bid Recommendation**:

- **URL**: `/bids/recommend/{item_id}`
- **Method**: `GET`
- **Response**:

```json
{
  "item_id": "550e8400-e29b-41d4-a716-446655440000",
  "recommended_bid": 55.00,
  "max_bid": 62.50,
  "confidence_score": 0.82,
  "profit_potential": 18.69,
  "roi_potential": 0.37,
  "risk_score": 0.25,
  "time_sensitivity": 0.75,
  "requires_review": false
}
```




**Schedule Bid**:

- **URL**: `/bids/schedule`
- **Method**: `POST`
- **Body**:

```json
{
  "item_id": "550e8400-e29b-41d4-a716-446655440000",
  "bid_amount": 55.00,
  "max_bid": 62.50,
  "scheduled_time": "2023-06-15T18:30:00Z"
}
```


- **Response**:

```json
{
  "bid_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "scheduled",
  "message": "Bid scheduled successfully"
}
```




#### Workflow Automation

**Run Daily Workflow**:

- **URL**: `/workflow/run_daily_workflow`
- **Method**: `POST`
- **Body**:

```json
{
  "workflow_id": "daily_full"
}
```


- **Response**:

```json
{
  "success": true,
  "message": "Workflow daily_full started",
  "data": {
    "workflow_id": "daily_full"
  }
}
```




**Get Workflow Status**:

- **URL**: `/workflow/get_workflow_status/{execution_id}`
- **Method**: `GET`
- **Response**:

```json
{
  "success": true,
  "message": "Workflow execution 550e8400-e29b-41d4-a716-446655440000 status",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "workflow_name": "Daily Full Workflow",
    "status": "COMPLETED",
    "start_time": "2023-06-15T01:00:00Z",
    "end_time": "2023-06-15T01:15:30Z",
    "tasks": [
      {
        "task_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
        "task_name": "scrape_hibid",
        "status": "COMPLETED",
        "start_time": "2023-06-15T01:00:05Z",
        "end_time": "2023-06-15T01:02:30Z"
      }
    ]
  }
}
```




**Fetch Recommendations**:

- **URL**: `/workflow/fetch_recommendations?limit=10&min_confidence=0.7&min_profit=20`
- **Method**: `GET`
- **Response**:

```json
{
  "success": true,
  "message": "Fetched 5 bid recommendations",
  "data": [
    {
      "item_id": "550e8400-e29b-41d4-a716-446655440000",
      "recommended_bid": 55.00,
      "max_bid": 62.50,
      "confidence_score": 0.82,
      "profit_potential": 18.69,
      "roi_potential": 0.37,
      "risk_score": 0.25,
      "time_sensitivity": 0.75,
      "requires_review": false
    }
  ]
}
```




#### Notifications

**Get Notifications**:

- **URL**: `/notifications?unread_only=true`
- **Method**: `GET`
- **Response**:

```json
{
  "notifications": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "type": "HIGH_PRIORITY_BID",
      "title": "Urgent Bid Opportunity",
      "message": "Urgent bid opportunity: Vintage Camera ending soon. Recommended bid: $55.00, potential profit: $18.69",
      "timestamp": "2023-06-15T12:30:00Z",
      "read": false,
      "priority": 5
    }
  ],
  "total": 1
}
```




**Mark Notification as Read**:

- **URL**: `/notifications/{notification_id}/read`
- **Method**: `POST`
- **Response**:

```json
{
  "success": true,
  "message": "Notification marked as read"
}
```