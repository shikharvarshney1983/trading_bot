# scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
import time
from app import app, update_live_prices, master_strategy_scheduler, scheduled_screener_job

def main():
    """
    Initializes and starts the APScheduler in its own process.
    It runs within the Flask app context to have access to the database and config.
    """
    with app.app_context():
        scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Kolkata')
        
        # Add jobs from your app.py
        scheduler.add_job(update_live_prices, 'cron', day_of_week='mon-fri', hour='9-16', minute='30')
        scheduler.add_job(master_strategy_scheduler, 'cron', day_of_week='mon-fri', hour=15, minute=0)
        scheduler.add_job(scheduled_screener_job, 'cron', day_of_week='fri', hour=18, minute=0)
        
        scheduler.start()
        print("Scheduler started successfully. Press Ctrl+C to exit.")

        # Keep the script alive
        try:
            while True:
                time.sleep(2)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
            print("Scheduler shut down.")

if __name__ == '__main__':
    main()
