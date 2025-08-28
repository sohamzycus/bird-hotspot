#!/usr/bin/env python3
"""
Configuration file for Cursor Usage Reporter
Customize this file with your specific email addresses and settings
"""

# Admin API Key for Cursor usage reporting
ADMIN_KEY = "key_5201f2de96e0dabe9607d5f4c753b6bb8844a2c5813996ccd426c99057472b4a"

# Email addresses to generate reports for
# Add or remove emails as needed
EMAIL_ADDRESSES = [
    "ram.samal@zycus.com",
    "anurag.singh@zycus.com",
    "samir.savasani@zycus.com",
    "rahul.prajapati@zycus.com",
    "alok.dubey@zycus.com",
    "manjunath.p@zycus.com",
    "naresh.r@zycus.com",
    "smrutiranjan.sahoo@zycus.com",
    "koppisetti.naren@zycus.com",
    "ram.mudumby@zycus.com",
    "chinthabotu.krishna@zycus.com",
    "veeru.garg@zycus.com",
    "durga.amulothu@zycus.com",
    "shital.gujarathi@zycus.com",
    "aditya.lokakshi@zycus.com",
    "uma.sai@zycus.com",
    "brajesh.kumar@zycus.com",
    "vineet.kapoor@zycus.com",
    "utkarsh.upadhyay@zycus.com",
    "shashishekar.j@zycus.com",
    "rahul.s@zycus.com",
    "divyansh.panwar@zycus.com",
    "shankhadeep.sarkar@zycus.com",
    # Add more emails here...
]

# Default number of past days to analyze
DEFAULT_PAST_DAYS = 7

# Default output format ("table", "json", "csv", "excel")
DEFAULT_OUTPUT_FORMAT = "table"

# Custom API base URL (Official Cursor API endpoint)
# The official Cursor Admin API endpoint is: https://api.cursor.com
# Reference: https://docs.cursor.com/en/account/teams/admin-api
# Leave as None to use the default, or specify a custom endpoint if needed
CUSTOM_API_BASE_URL = None  # Uses https://api.cursor.com by default

# Report settings
REPORT_SETTINGS = {
    "include_summary": True,
    "save_to_file": True,
    "show_timestamps": True,
    "timezone": "UTC"
}
