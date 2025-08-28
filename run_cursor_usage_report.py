#!/usr/bin/env python3
"""
Simple script to run Cursor usage reports using the configuration file
"""

import sys
from cursor_usage_reporter import CursorUsageReporter
from cursor_usage_config import (
    ADMIN_KEY, 
    EMAIL_ADDRESSES, 
    DEFAULT_PAST_DAYS,
    DEFAULT_OUTPUT_FORMAT,
    CUSTOM_API_BASE_URL
)

def run_usage_report(emails=None, past_days=None, output_format=None):
    """
    Run cursor usage report with configurable parameters.
    
    Parameters:
    -----------
    emails : list, optional
        List of email addresses (uses config default if None)
    past_days : int, optional
        Number of past days (uses config default if None)
    output_format : str, optional
        Output format (uses config default if None)
    """
    
    # Use defaults from config if not provided
    emails = emails or EMAIL_ADDRESSES
    past_days = past_days or DEFAULT_PAST_DAYS
    output_format = output_format or DEFAULT_OUTPUT_FORMAT
    
    if not emails:
        print("❌ No email addresses configured. Please update cursor_usage_config.py")
        return False
    
    try:
        print(f"🚀 Generating Cursor usage report for {len(emails)} users...")
        print(f"📅 Period: Last {past_days} days")
        print(f"📊 Format: {output_format}")
        print("-" * 50)
        
        # Initialize reporter with custom API base URL if provided
        reporter = CursorUsageReporter(ADMIN_KEY, CUSTOM_API_BASE_URL)
        
        # Generate and display report
        report = reporter.generate_report(emails, past_days, output_format)
        print(report)
        
        # Save to files in all formats or specified format
        if output_format == "table":
            table_file = reporter.save_report_to_file(emails, past_days, "table")
            json_file = reporter.save_report_to_file(emails, past_days, "json")
            csv_file = reporter.save_report_to_file(emails, past_days, "csv")
            excel_file = reporter.save_report_to_file(emails, past_days, "excel")
            
            print(f"\n📄 Reports saved:")
            print(f"   📋 Table: {table_file}")
            print(f"   📋 JSON:  {json_file}")
            print(f"   📋 CSV:   {csv_file}")
            print(f"   📊 Excel: {excel_file}")
        else:
            saved_file = reporter.save_report_to_file(emails, past_days, output_format)
            print(f"\n📄 Report saved: {saved_file}")
        
        print("\n✅ Report generation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return False

def main():
    """Main function with command line argument support."""
    
    if len(sys.argv) > 1:
        try:
            # Parse command line arguments
            past_days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PAST_DAYS
            output_format = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FORMAT
            
            print(f"📋 Using command line arguments:")
            print(f"   Past days: {past_days}")
            print(f"   Output format: {output_format}")
            
            success = run_usage_report(past_days=past_days, output_format=output_format)
            
        except ValueError:
            print("❌ Invalid arguments. Usage: python run_cursor_usage_report.py [past_days] [output_format]")
            print("   Examples:")
            print("     python run_cursor_usage_report.py 14 json")
            print("     python run_cursor_usage_report.py 7 excel")
            print("     python run_cursor_usage_report.py 30 csv")
            sys.exit(1)
    else:
        # Use configuration defaults
        print("📋 Using configuration defaults from cursor_usage_config.py")
        success = run_usage_report()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
