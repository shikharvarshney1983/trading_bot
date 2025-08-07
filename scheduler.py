# scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
import time
from app import app, update_live_prices, master_scheduler_sell_add, scheduled_screener_job, master_scheduler_watchlist, master_scheduler_buy

def main():
    """
    Initializes and starts the APScheduler in its own process.
    It runs within the Flask app context to have access to the database and config.
    """
    with app.app_context():
        scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Kolkata')
        
        # Add jobs from your app.py
        scheduler.add_job(update_live_prices, 'cron', day_of_week='mon-fri', hour='9-16', minute='30')
        
        # 1. Execute SELL and ADD-ON trades at 3:00 PM on weekdays
        scheduler.add_job(master_scheduler_sell_add, 'cron', day_of_week='mon-fri', hour=15, minute=0)

        # 2. Generate user-specific watchlists at 4:00 PM on weekdays
        scheduler.add_job(master_scheduler_watchlist, 'cron', day_of_week='mon-fri', hour=16, minute=0)

        # 3. Execute BUY trades from user's list at 9:15 AM on weekdays
        scheduler.add_job(master_scheduler_buy, 'cron', day_of_week='mon-fri', hour=9, minute=15)
        
        # Updated screener jobs
        # Daily runs after market close on weekdays
        scheduler.add_job(scheduled_screener_job, 'cron', day_of_week='mon-fri', hour=16, minute=0, args=['daily'])
        
        # Weekly runs after market close on Friday
        scheduler.add_job(scheduled_screener_job, 'cron', day_of_week='fri', hour=17, minute=0, args=['weekly'])
        
        # Monthly runs after market close on the last Friday of the month
        # CORRECTED: Use day='last fri' instead of day_of_week
        scheduler.add_job(scheduled_screener_job, 'cron', day='last fri', hour=18, minute=0, args=['monthly'])
        
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