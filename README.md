# Flask Paper Trading Bot & Stock Screener

This is a web application built with Flask that allows users to paper trade stocks based on a quantitative trading strategy. The application includes a user dashboard, portfolio tracking, and a stock screener to identify trading opportunities based on technical indicators.

## Features

* **User Authentication**: Secure login system with separate roles for 'Admin' and 'User'.
* **Admin Panel**: A dedicated interface for administrators to create, manage, and delete user accounts, as well as update user passwords and balances.
* **User Dashboard**: A comprehensive dashboard for users to view key performance indicators like portfolio value, unrealized/realized P/L, and win ratio. It also displays the current portfolio, transaction history, and a P/L calendar.
* **Strategy Configuration**: Users can configure their own trading strategy parameters, including base capital, stock lists, brokerage fees, and position sizing.
* **Automated & Manual Trading**: The trading bot can be run manually or scheduled to execute automatically, performing buy/sell actions based on a defined EMA crossover and momentum strategy.
* **Stock Screener**: An admin-accessible tool to scan a master list of stocks against technical criteria like ADX, RSI, and EMA positioning to find stocks that meet the strategy's entry conditions. Results can be downloaded as an Excel file.
* **Background Task Processing**: Long-running tasks, such as running the screener, are executed in background threads to prevent UI timeouts and provide a responsive user experience. The UI polls the server to get a notification when the task is complete.
* **Scheduled Jobs**: Uses APScheduler to run the trading strategy and update market data automatically at predefined times. The scheduler is designed to run as a separate, dedicated process for production stability.

## Technology Stack

* **Backend**: Flask, Gunicorn
* **Data & Analysis**: pandas, yfinance, pandas-ta
* **Scheduling**: APScheduler
* **Database**: SQLite
* **Frontend**: Bootstrap 5, jQuery, DataTables

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd trading_bot
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # For Linux/macOS
    python3 -m venv tbenv
    source tbenv/bin/activate

    # For Windows
    py -m venv tbenv
    tbenv\Scripts\activate
    ```

3.  **Install Dependencies**
    *(Create a `requirements.txt` file if you don't have one: `pip freeze > requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add the following keys:
    ```
    FLASK_APP=app.py
    SECRET_KEY='a-very-strong-and-secret-key'
    TELEGRAM_BOT_TOKEN='your_telegram_bot_token_here' # Optional
    ```

5.  **Initialize the Database**
    Run the custom `init-db` command. This will create the `trading.db` file with all the necessary tables and create a default admin user.
    ```bash
    flask init-db
    ```
    The default admin credentials are:
    * **Username**: `admin`
    * **Password**: `admin`

## Usage

The application is designed to run as two separate processes in production: one for the web server and one for the scheduler.

1.  **Start the Web Application**
    To run the Flask development server, use the following command. The `use_reloader=False` flag is recommended to prevent issues with background threads and the scheduler during development.
    ```python
    # In your app.py, ensure the main block looks like this:
    if __name__ == '__main__':
        app.run(debug=True, port=5000, use_reloader=False)
    ```
    For production, use a WSGI server like Gunicorn:
    ```bash
    gunicorn --workers 3 --bind 0.0.0.0:5000 app:app
    ```

2.  **Start the Scheduler**
    In a **separate terminal**, run the scheduler script:
    ```bash
    python scheduler.py
    ```

3.  **Access the Application**
    Open your web browser and navigate to `http://127.0.0.1:5000`. You can log in using the default admin credentials to access all features or create new users via the admin panel.
