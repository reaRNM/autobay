## Installation and Setup

### System Requirements

- **Operating System**: Linux (Debian Trixie on Chromebook)
- **Python**: Version 3.8 or higher
- **PostgreSQL**: Version 13 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: Minimum 10GB free space
- **Internet Connection**: Required for data collection and API access


### Prerequisites

1. **Python 3.8+**:

```shellscript
sudo apt update
sudo apt install python3 python3-pip python3-venv
```


2. **PostgreSQL**:

```shellscript
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```


3. **Node.js and npm** (for Dashboard):

```shellscript
sudo apt install nodejs npm
```


4. **Git**:

```shellscript
sudo apt install git
```


5. **Required System Packages**:

```shellscript
sudo apt install build-essential libpq-dev python3-dev
```




### Installation Steps

1. **Clone the Repository**:

```shellscript
git clone https://github.com/yourusername/autobay.git
cd autobay
```


2. **Create and Activate Virtual Environment**:

```shellscript
python3 -m venv venv
source venv/bin/activate
```


3. **Install Python Dependencies**:

```shellscript
pip install -r requirements.txt
```


4. **Set Up PostgreSQL Database**:

```shellscript
sudo -u postgres psql
```

In the PostgreSQL prompt:

```sql
CREATE DATABASE autobay;
CREATE USER autobay_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE autobay TO autobay_user;
\q
```


5. **Initialize Database Schema**:

```shellscript
python scripts/init_db.py
```


6. **Configure Environment Variables**:
Create a `.env` file in the project root:

```plaintext
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=autobay
DB_USER=autobay_user
DB_PASSWORD=your_password

# API Keys
EBAY_API_KEY=your_ebay_api_key
AMAZON_API_KEY=your_amazon_api_key

# Notification Services
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
FIREBASE_API_KEY=your_firebase_api_key

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
SECRET_KEY=your_secret_key
```


7. **Build the Dashboard Frontend** (if applicable):

```shellscript
cd dashboard
npm install
npm run build
cd ..
```


8. **Create Log Directory**:

```shellscript
mkdir -p logs
```