#!/usr/bin/env python3
"""
Cursor Usage Reporter - Fetch usage reports for given emails and time periods
"""

import requests
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Optional, Union
import os
import sys
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cursor_usage.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cursor_usage")

@dataclass
class UsageMetrics:
    """Data class to hold usage metrics for a user."""
    email: str
    total_completions: int
    total_chats: int
    total_commands: int
    total_time_hours: float
    last_activity: str
    subscription_type: str

class CursorUsageReporter:
    """Client for fetching Cursor usage reports via admin API."""
    
    def __init__(self, admin_key: str, base_url: str = None):
        """
        Initialize the Cursor Usage Reporter.
        
        Parameters:
        -----------
        admin_key : str
            Admin API key for Cursor usage reporting (format: key_xxxxx...)
        base_url : str, optional
            Base URL for the Cursor API (default: https://api.cursor.com)
        """
        self.admin_key = admin_key
        self.base_url = base_url or "https://api.cursor.com"  # Official Cursor API endpoint
        self.session = requests.Session()
        
        # Use basic authentication as per Cursor API documentation
        # API key is used as username, password is empty
        self.session.auth = (admin_key, "")
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "CursorUsageReporter/1.0"
        })
    
    def fetch_usage_report(self, 
                          emails: Union[str, List[str]], 
                          past_days: int = 7) -> List[UsageMetrics]:
        """
        Fetch usage reports for specified emails and time period.
        
        Parameters:
        -----------
        emails : str or List[str]
            Single email address or list of email addresses (or None for all team members)
        past_days : int
            Number of past days to fetch data for (default: 7, max: 90)
            
        Returns:
        --------
        List[UsageMetrics]
            List of usage metrics for each user
        """
        # Normalize emails to list
        if isinstance(emails, str):
            emails = [emails]
        
        # Calculate date range (Cursor API uses epoch milliseconds)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=past_days)
        start_epoch = int(start_date.timestamp() * 1000)
        end_epoch = int(end_date.timestamp() * 1000)
        
        logger.info(f"Fetching usage data from {start_date.date()} to {end_date.date()} (past {past_days} days)")
        
        # Fetch daily usage data from Cursor API
        daily_usage_data = self._fetch_daily_usage_data(start_epoch, end_epoch)
        
        if not daily_usage_data:
            logger.warning("No daily usage data returned from API")
            return []
        
        # Filter by emails if specified, otherwise include all users
        if emails:
            filtered_data = [entry for entry in daily_usage_data if entry.get('email') in emails]
            logger.info(f"Filtered data to {len(filtered_data)} entries for specified emails")
        else:
            filtered_data = daily_usage_data
            logger.info(f"Processing data for all {len(filtered_data)} user entries")
        
        # Aggregate usage metrics per user
        usage_data = self._aggregate_usage_metrics(filtered_data, emails or [])
        
        logger.info(f"✅ Successfully processed usage data for {len(usage_data)} users")
        return usage_data
    
    def _fetch_daily_usage_data(self, start_epoch: int, end_epoch: int) -> List[Dict]:
        """
        Fetch daily usage data from Cursor API.
        
        Parameters:
        -----------
        start_epoch : int
            Start date in epoch milliseconds
        end_epoch : int
            End date in epoch milliseconds
            
        Returns:
        --------
        List[Dict]
            List of daily usage data entries
        """
        url = f"{self.base_url}/teams/daily-usage-data"
        
        # Request body as per Cursor API documentation
        request_body = {
            "startDate": start_epoch,
            "endDate": end_epoch
        }
        
        try:
            logger.info(f"Making API request to {url}")
            response = self.session.post(url, json=request_body, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                daily_data = data.get('data', [])
                logger.info(f"✅ Successfully fetched {len(daily_data)} daily usage entries")
                return daily_data
            elif response.status_code == 401:
                logger.error("❌ Unauthorized - check admin key format and permissions")
                logger.error(f"API Key format: {self.admin_key[:20]}...")
                return []
            elif response.status_code == 403:
                logger.error("❌ Forbidden - admin key may not have required permissions")
                return []
            elif response.status_code == 429:
                logger.warning("⚠️ Rate limit exceeded - waiting before retry...")
                import time
                time.sleep(5)
                return self._fetch_daily_usage_data(start_epoch, end_epoch)
            else:
                logger.error(f"❌ API request failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"❌ Network error: {str(e)}")
            return []
    
    def _aggregate_usage_metrics(self, daily_data: List[Dict], target_emails: List[str] = None) -> List[UsageMetrics]:
        """
        Aggregate daily usage data into per-user metrics.
        
        Parameters:
        -----------
        daily_data : List[Dict]
            List of daily usage data entries from Cursor API
        target_emails : List[str], optional
            List of emails to filter for (None for all users)
            
        Returns:
        --------
        List[UsageMetrics]
            List of aggregated usage metrics per user
        """
        user_aggregates = {}
        
        for entry in daily_data:
            email = entry.get('email')
            if not email:
                continue
                
            # Filter by target emails if specified
            if target_emails and email not in target_emails:
                continue
            
            # Initialize user aggregate if not exists
            if email not in user_aggregates:
                user_aggregates[email] = {
                    'total_completions': 0,
                    'total_chats': 0,
                    'total_commands': 0,
                    'total_applies': 0,
                    'total_accepts': 0,
                    'total_rejects': 0,
                    'total_tabs_shown': 0,
                    'total_tabs_accepted': 0,
                    'composer_requests': 0,
                    'chat_requests': 0,
                    'agent_requests': 0,
                    'cmdK_usages': 0,
                    'active_days': 0,
                    'total_lines_added': 0,
                    'total_lines_deleted': 0,
                    'accepted_lines_added': 0,
                    'accepted_lines_deleted': 0,
                    'subscription_included_reqs': 0,
                    'usage_based_reqs': 0,
                    'bugbot_usages': 0,
                    'last_activity': None,
                    'most_used_models': []
                }
            
            user_data = user_aggregates[email]
            
            # Aggregate the metrics
            user_data['total_applies'] += entry.get('totalApplies', 0)
            user_data['total_accepts'] += entry.get('totalAccepts', 0)
            user_data['total_rejects'] += entry.get('totalRejects', 0)
            user_data['total_tabs_shown'] += entry.get('totalTabsShown', 0)
            user_data['total_tabs_accepted'] += entry.get('totalTabsAccepted', 0)
            user_data['composer_requests'] += entry.get('composerRequests', 0)
            user_data['chat_requests'] += entry.get('chatRequests', 0)
            user_data['agent_requests'] += entry.get('agentRequests', 0)
            user_data['cmdK_usages'] += entry.get('cmdkUsages', 0)
            user_data['total_lines_added'] += entry.get('totalLinesAdded', 0)
            user_data['total_lines_deleted'] += entry.get('totalLinesDeleted', 0)
            user_data['accepted_lines_added'] += entry.get('acceptedLinesAdded', 0)
            user_data['accepted_lines_deleted'] += entry.get('acceptedLinesDeleted', 0)
            user_data['subscription_included_reqs'] += entry.get('subscriptionIncludedReqs', 0)
            user_data['usage_based_reqs'] += entry.get('usageBasedReqs', 0)
            user_data['bugbot_usages'] += entry.get('bugbotUsages', 0)
            
            # Count active days
            if entry.get('isActive', False):
                user_data['active_days'] += 1
            
            # Track most recent activity
            entry_date = entry.get('date')
            if entry_date and (user_data['last_activity'] is None or entry_date > user_data['last_activity']):
                user_data['last_activity'] = entry_date
            
            # Track models used
            model = entry.get('mostUsedModel')
            if model:
                user_data['most_used_models'].append(model)
        
        # Convert to UsageMetrics objects
        result = []
        for email, data in user_aggregates.items():
            # Calculate total completions (combines applies and tab completions)
            total_completions = data['total_applies'] + data['total_tabs_accepted']
            
            # Calculate total chats (composer + chat + agent requests)
            total_chats = data['composer_requests'] + data['chat_requests'] + data['agent_requests']
            
            # Calculate total commands (cmdK + other operations)
            total_commands = data['cmdK_usages'] + data['total_applies']
            
            # Format last activity
            last_activity = "N/A"
            if data['last_activity']:
                try:
                    last_activity_dt = datetime.fromtimestamp(data['last_activity'] / 1000)
                    last_activity = last_activity_dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    last_activity = str(data['last_activity'])
            
            # Determine subscription type (simplified)
            if data['usage_based_reqs'] > 0:
                subscription_type = "usage-based"
            elif data['subscription_included_reqs'] > 0:
                subscription_type = "business/pro"
            else:
                subscription_type = "free"
            
            # Calculate approximate active hours (estimate: 8 hours per active day)
            estimated_hours = data['active_days'] * 8.0
            
            result.append(UsageMetrics(
                email=email,
                total_completions=total_completions,
                total_chats=total_chats,
                total_commands=total_commands,
                total_time_hours=estimated_hours,
                last_activity=last_activity,
                subscription_type=subscription_type
            ))
        
        return result
    
    def generate_report(self, 
                       emails: Union[str, List[str]], 
                       past_days: int = 7,
                       output_format: str = "table") -> str:
        """
        Generate a formatted usage report.
        
        Parameters:
        -----------
        emails : str or List[str]
            Email addresses to generate report for
        past_days : int
            Number of past days to include
        output_format : str
            Format for output ("table", "json", "csv", "excel")
            
        Returns:
        --------
        str
            Formatted report (or file path for Excel format)
        """
        usage_data = self.fetch_usage_report(emails, past_days)
        
        if not usage_data:
            return "No usage data found for the specified users."
        
        if output_format.lower() == "json":
            return self._format_json_report(usage_data, past_days)
        elif output_format.lower() == "csv":
            return self._format_csv_report(usage_data, past_days)
        elif output_format.lower() == "excel":
            return self._format_excel_report(usage_data, past_days)
        else:
            return self._format_table_report(usage_data, past_days)
    
    def _format_table_report(self, usage_data: List[UsageMetrics], past_days: int) -> str:
        """Format usage data as a readable table."""
        report = f"\n📊 CURSOR USAGE REPORT - LAST {past_days} DAYS\n"
        report += "=" * 80 + "\n\n"
        
        for metrics in usage_data:
            report += f"👤 USER: {metrics.email}\n"
            report += f"   💬 Completions: {metrics.total_completions:,}\n"
            report += f"   🗨️  Chats: {metrics.total_chats:,}\n"
            report += f"   ⚡ Commands: {metrics.total_commands:,}\n"
            report += f"   ⏱️  Active Hours: {metrics.total_time_hours:.1f}\n"
            report += f"   📅 Last Activity: {metrics.last_activity}\n"
            report += f"   💳 Subscription: {metrics.subscription_type}\n"
            report += "-" * 50 + "\n"
        
        # Summary
        total_completions = sum(m.total_completions for m in usage_data)
        total_chats = sum(m.total_chats for m in usage_data)
        total_commands = sum(m.total_commands for m in usage_data)
        total_hours = sum(m.total_time_hours for m in usage_data)
        
        report += f"\n📈 SUMMARY ({len(usage_data)} users):\n"
        report += f"   Total Completions: {total_completions:,}\n"
        report += f"   Total Chats: {total_chats:,}\n"
        report += f"   Total Commands: {total_commands:,}\n"
        report += f"   Total Active Hours: {total_hours:.1f}\n"
        
        return report
    
    def _format_json_report(self, usage_data: List[UsageMetrics], past_days: int) -> str:
        """Format usage data as JSON."""
        report_data = {
            "report_period_days": past_days,
            "generated_at": datetime.now().isoformat(),
            "users": [
                {
                    "email": m.email,
                    "completions": m.total_completions,
                    "chats": m.total_chats,
                    "commands": m.total_commands,
                    "active_hours": m.total_time_hours,
                    "last_activity": m.last_activity,
                    "subscription_type": m.subscription_type
                }
                for m in usage_data
            ]
        }
        return json.dumps(report_data, indent=2)
    
    def _format_csv_report(self, usage_data: List[UsageMetrics], past_days: int) -> str:
        """Format usage data as CSV."""
        df = pd.DataFrame([
            {
                "Email": m.email,
                "Completions": m.total_completions,
                "Chats": m.total_chats,
                "Commands": m.total_commands,
                "Active_Hours": m.total_time_hours,
                "Last_Activity": m.last_activity,
                "Subscription_Type": m.subscription_type
            }
            for m in usage_data
        ])
        return df.to_csv(index=False)
    
    def _format_excel_report(self, usage_data: List[UsageMetrics], past_days: int) -> str:
        """Format usage data as Excel file and return the filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cursor_usage_report_{timestamp}.xlsx"
        
        # Create DataFrame with detailed metrics
        df = pd.DataFrame([
            {
                "User Email": m.email,
                "Total Completions": m.total_completions,
                "Total Chats": m.total_chats,
                "Total Commands": m.total_commands,
                "Active Hours (Est.)": m.total_time_hours,
                "Last Activity": m.last_activity,
                "Subscription Type": m.subscription_type
            }
            for m in usage_data
        ])
        
        # Calculate summary statistics
        total_completions = sum(m.total_completions for m in usage_data)
        total_chats = sum(m.total_chats for m in usage_data)
        total_commands = sum(m.total_commands for m in usage_data)
        total_hours = sum(m.total_time_hours for m in usage_data)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame([
            ["Report Period (Days)", past_days],
            ["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Total Users", len(usage_data)],
            ["Total Completions", total_completions],
            ["Total Chats", total_chats],
            ["Total Commands", total_commands],
            ["Total Active Hours", f"{total_hours:.1f}"],
            ["Avg Completions per User", f"{total_completions/len(usage_data):.1f}" if usage_data else "0"],
            ["Avg Chats per User", f"{total_chats/len(usage_data):.1f}" if usage_data else "0"],
            ["Avg Hours per User", f"{total_hours/len(usage_data):.1f}" if usage_data else "0"]
        ], columns=["Metric", "Value"])
        
        # Write to Excel with multiple sheets
        try:
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                # Write main data
                df.to_excel(writer, sheet_name='User Details', index=False)
                
                # Write summary
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Get workbook and worksheet objects
                workbook = writer.book
                worksheet1 = writer.sheets['User Details']
                worksheet2 = writer.sheets['Summary']
                
                # Add formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                number_format = workbook.add_format({'num_format': '#,##0'})
                hours_format = workbook.add_format({'num_format': '#,##0.0'})
                
                # Format headers for User Details sheet
                for col_num, value in enumerate(df.columns.values):
                    worksheet1.write(0, col_num, value, header_format)
                
                # Format data columns in User Details sheet
                worksheet1.set_column('A:A', 30)  # Email column wider
                worksheet1.set_column('B:D', 12, number_format)  # Numeric columns
                worksheet1.set_column('E:E', 15, hours_format)   # Hours column
                worksheet1.set_column('F:F', 20)  # Last activity
                worksheet1.set_column('G:G', 15)  # Subscription type
                
                # Format Summary sheet
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet2.write(0, col_num, value, header_format)
                worksheet2.set_column('A:A', 25)
                worksheet2.set_column('B:B', 20)
                
                # Add charts to Summary sheet
                # Create a chart for completions by user
                chart = workbook.add_chart({'type': 'column'})
                chart.add_series({
                    'name': 'Completions by User',
                    'categories': ['User Details', 1, 0, len(usage_data), 0],  # Email column
                    'values': ['User Details', 1, 1, len(usage_data), 1],      # Completions column
                })
                chart.set_title({'name': 'Completions by User'})
                chart.set_x_axis({'name': 'Users'})
                chart.set_y_axis({'name': 'Completions'})
                chart.set_size({'width': 600, 'height': 400})
                
                # Insert chart in Summary sheet
                worksheet2.insert_chart('D2', chart)
                
            logger.info(f"📊 Excel report saved to: {filename}")
            return f"Excel report saved to: {filename}"
            
        except Exception as e:
            logger.error(f"❌ Failed to create Excel report: {str(e)}")
            return f"Error creating Excel report: {str(e)}"
    
    def save_report_to_file(self, 
                           emails: Union[str, List[str]], 
                           past_days: int = 7,
                           output_format: str = "table",
                           filename: Optional[str] = None) -> str:
        """
        Generate and save report to file.
        
        Parameters:
        -----------
        emails : str or List[str]
            Email addresses to generate report for
        past_days : int
            Number of past days to include
        output_format : str
            Format for output ("table", "json", "csv", "excel")
        filename : Optional[str]
            Custom filename (auto-generated if None)
            
        Returns:
        --------
        str
            Path to saved file
        """
        # For Excel format, the file is created directly in _format_excel_report
        if output_format.lower() == "excel":
            report_result = self.generate_report(emails, past_days, output_format)
            # Extract filename from the result message
            if "Excel report saved to:" in report_result:
                return report_result.split("Excel report saved to: ")[1]
            else:
                return report_result  # Return error message if failed
        
        # For other formats, generate report content and save to file
        report = self.generate_report(emails, past_days, output_format)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "txt" if output_format == "table" else output_format
            filename = f"cursor_usage_report_{timestamp}.{extension}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {filename}")
        return filename

def main():
    """Example usage of the Cursor Usage Reporter."""
    
    # Configuration
    ADMIN_KEY = "key_5201f2de96e0dabe9607d5f4c753b6bb8844a2c5813996ccd426c99057472b4a"
    
    # Example email list - modify as needed
    emails = [
        "user1@example.com",
        "user2@example.com",
        "user3@example.com"
    ]
    
    # Number of past days to analyze
    past_days = 7
    
    try:
        # Initialize the reporter with official Cursor API endpoint
        reporter = CursorUsageReporter(ADMIN_KEY)
        
        print("🚀 Starting Cursor Usage Report Generation...")
        
        # Generate and display table report
        print(reporter.generate_report(emails, past_days, "table"))
        
        # Save detailed reports in different formats
        reporter.save_report_to_file(emails, past_days, "table")
        reporter.save_report_to_file(emails, past_days, "json")
        reporter.save_report_to_file(emails, past_days, "csv")
        reporter.save_report_to_file(emails, past_days, "excel")
        
        print("\n✅ All reports generated successfully!")
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
