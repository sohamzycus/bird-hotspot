# Cursor Usage Reporter

A Python tool to fetch and analyze Cursor usage reports for multiple users over configurable time periods.

## 📁 Files Overview

- **`cursor_usage_reporter.py`** - Main reporter class with full API functionality
- **`cursor_usage_config.py`** - Configuration file for emails and settings
- **`run_cursor_usage_report.py`** - Simple script to run reports easily

## 🚀 Quick Start

### 1. Configure Email Addresses

Edit `cursor_usage_config.py` to add the email addresses you want to track:

```python
EMAIL_ADDRESSES = [
    "user1@company.com",
    "user2@company.com", 
    "user3@company.com",
    # Add more emails here...
]
```

### 2. Run the Report

**Basic usage (uses config defaults):**
```bash
python run_cursor_usage_report.py
```

**Custom time period:**
```bash
python run_cursor_usage_report.py 14        # Last 14 days
```

**Custom format:**
```bash
python run_cursor_usage_report.py 7 json    # Last 7 days, JSON format
python run_cursor_usage_report.py 30 csv    # Last 30 days, CSV format
```

## 📊 Report Formats

### Table Format (Default)
```
📊 CURSOR USAGE REPORT - LAST 7 DAYS
================================================================================

👤 USER: user1@company.com
   💬 Completions: 1,234
   🗨️  Chats: 89
   ⚡ Commands: 456
   ⏱️  Active Hours: 12.5
   📅 Last Activity: 2024-01-15T14:30:00Z
   💳 Subscription: pro
--------------------------------------------------

📈 SUMMARY (3 users):
   Total Completions: 3,456
   Total Chats: 234
   Total Commands: 1,234
   Total Active Hours: 35.2
```

### JSON Format
```json
{
  "report_period_days": 7,
  "generated_at": "2024-01-15T15:00:00Z",
  "users": [
    {
      "email": "user1@company.com",
      "completions": 1234,
      "chats": 89,
      "commands": 456,
      "active_hours": 12.5,
      "last_activity": "2024-01-15T14:30:00Z",
      "subscription_type": "pro"
    }
  ]
}
```

### CSV Format
```csv
Email,Completions,Chats,Commands,Active_Hours,Last_Activity,Subscription_Type
user1@company.com,1234,89,456,12.5,2024-01-15T14:30:00Z,pro
```

## ⚙️ Advanced Usage

### Direct API Usage

```python
from cursor_usage_reporter import CursorUsageReporter

# Initialize with admin key
reporter = CursorUsageReporter("your_admin_key_here")

# Single user, custom time period
metrics = reporter.fetch_usage_report("user@company.com", past_days=14)

# Multiple users
emails = ["user1@company.com", "user2@company.com"]
metrics_list = reporter.fetch_usage_report(emails, past_days=30)

# Generate custom report
report = reporter.generate_report(emails, past_days=7, output_format="json")
print(report)

# Save to file with custom name
reporter.save_report_to_file(emails, 7, "csv", "weekly_usage.csv")
```

### Configuration Options

Edit `cursor_usage_config.py` to customize:

```python
# Admin API Key
ADMIN_KEY = "your_admin_key_here"

# Default settings
DEFAULT_PAST_DAYS = 7
DEFAULT_OUTPUT_FORMAT = "table"  # "table", "json", or "csv"

# Custom API endpoint (if needed)
CUSTOM_API_BASE_URL = "https://your-custom-api.com"
```

## 📈 Usage Metrics Explained

- **Completions**: Number of AI code completions generated
- **Chats**: Number of chat conversations with Cursor AI
- **Commands**: Number of Cursor commands executed
- **Active Hours**: Total time actively using Cursor
- **Last Activity**: Timestamp of most recent activity
- **Subscription Type**: User's subscription level (free, pro, etc.)

## 🛠️ Error Handling

The reporter includes comprehensive error handling:

- **Invalid API Key**: Returns 403 error with helpful message
- **User Not Found**: Logs warning and continues with other users
- **Rate Limiting**: Automatically retries with delay
- **Network Issues**: Graceful failure with error logging

## 📝 Logging

All operations are logged to `cursor_usage.log` with timestamps and severity levels:

```
2024-01-15 15:00:00 - cursor_usage - INFO - Fetching usage data for 3 users from 2024-01-08 to 2024-01-15
2024-01-15 15:00:01 - cursor_usage - INFO - ✅ Successfully fetched data for user1@company.com
2024-01-15 15:00:02 - cursor_usage - WARNING - ⚠️ No data found for user2@company.com
```

## 🔒 Security Notes

- Admin key is stored in configuration file - keep it secure
- Never commit the admin key to version control
- Consider using environment variables for production usage:

```python
import os
ADMIN_KEY = os.getenv("CURSOR_ADMIN_KEY", "fallback_key")
```

## 🆘 Troubleshooting

### Common Issues

1. **"Access denied" error**
   - Check admin key is correct and has proper permissions

2. **"User not found" warnings**
   - Verify email addresses are exact matches in Cursor system

3. **Empty reports**
   - Check date range and verify users had activity in that period

4. **Rate limiting**
   - Script automatically handles this with delays between requests

### Debug Mode

Enable debug logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📋 Requirements

All required packages are already listed in `requirements.txt`:
- `requests>=2.31.0`
- `pandas>=1.5.3`
- Standard library modules (json, datetime, logging, etc.)

Install with:
```bash
pip install -r requirements.txt
```

## 🤝 Support

For issues or questions about the usage reporter, check the log files for detailed error messages and consult the API documentation for your Cursor admin portal.
