## Running the Program

### Starting the Backend

1. **Activate the Virtual Environment** (if not already activated):

```shellscript
source venv/bin/activate
```


2. **Start the API Server**:

```shellscript
python -m autobay.api
```

The API server will start on `http://localhost:8000` by default.




### Starting the Dashboard

1. **Development Mode**:

```shellscript
cd dashboard
npm start
```

The dashboard will be available at `http://localhost:3000`.


2. **Production Mode**:
If you've built the dashboard (step 7 in Installation), it will be served by the API server at `http://localhost:8000`.


### Running Modules Independently

#### Profit Calculator

```shellscript
python -m autobay.profit_calculator --item-id=12345 --platform=ebay
```

#### Bid Intelligence

```shellscript
python -m autobay.bid_intelligence --budget=1000 --category=electronics
```

#### Data Collection

```shellscript
python -m autobay.scrapers.hibid --categories=electronics,collectibles --max-pages=5
```

#### Daily Workflow

```shellscript
python -m autobay.workflow_automation.example
```