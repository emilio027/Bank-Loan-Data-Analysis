# power_bi_integration.py
# Power BI REST API Integration Template
# Requires env: PBI_TENANT_ID, PBI_CLIENT_ID, PBI_CLIENT_SECRET, PBI_GROUP_ID, PBI_DATASET_ID

import os
import time
import requests
import json
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PowerBIIntegration:
    """
    Power BI REST API integration for automated dataset refresh and report export.
    
    Environment Variables Required:
    - PBI_TENANT_ID: Azure AD tenant ID
    - PBI_CLIENT_ID: Power BI app client ID  
    - PBI_CLIENT_SECRET: Power BI app client secret
    - PBI_GROUP_ID: Power BI workspace ID
    - PBI_DATASET_ID: Power BI dataset ID
    """
    
    def __init__(self):
        self.tenant_id = os.getenv("PBI_TENANT_ID")
        self.client_id = os.getenv("PBI_CLIENT_ID")
        self.client_secret = os.getenv("PBI_CLIENT_SECRET")
        self.group_id = os.getenv("PBI_GROUP_ID")
        self.dataset_id = os.getenv("PBI_DATASET_ID")
        
        if not all([self.tenant_id, self.client_id, self.client_secret]):
            logger.warning("Power BI credentials not fully configured. Check environment variables.")
    
    def get_access_token(self) -> Optional[str]:
        """Get Azure AD access token for Power BI API."""
        try:
            url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "https://analysis.windows.net/powerbi/api/.default",
                "grant_type": "client_credentials",
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            return response.json()["access_token"]
            
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            return None
    
    def refresh_dataset(self) -> bool:
        """Trigger dataset refresh in Power BI."""
        try:
            token = self.get_access_token()
            if not token:
                return False
            
            url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}/datasets/{self.dataset_id}/refreshes"
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.post(url, headers=headers, json={})
            response.raise_for_status()
            
            logger.info("Dataset refresh triggered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh dataset: {e}")
            return False
    
    def get_refresh_status(self) -> Optional[Dict[str, Any]]:
        """Check status of latest dataset refresh."""
        try:
            token = self.get_access_token()
            if not token:
                return None
            
            url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}/datasets/{self.dataset_id}/refreshes"
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            refreshes = response.json()["value"]
            if refreshes:
                return refreshes[0]  # Latest refresh
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get refresh status: {e}")
            return None
    
    def wait_for_refresh_completion(self, timeout_minutes: int = 30) -> bool:
        """Wait for dataset refresh to complete."""
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            status = self.get_refresh_status()
            if status:
                if status["status"] == "Completed":
                    logger.info("Dataset refresh completed successfully")
                    return True
                elif status["status"] == "Failed":
                    logger.error(f"Dataset refresh failed: {status.get('serviceExceptionJson', '')}")
                    return False
            
            time.sleep(30)  # Check every 30 seconds
        
        logger.error("Dataset refresh timeout")
        return False
    
    def export_report_to_file(self, report_id: str, format: str = "PDF") -> Optional[bytes]:
        """Export Power BI report to file (PDF, PNG, etc.)."""
        try:
            token = self.get_access_token()
            if not token:
                return None
            
            # Start export
            export_url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}/reports/{report_id}/ExportTo"
            headers = {"Authorization": f"Bearer {token}"}
            data = {"format": format}
            
            response = requests.post(export_url, headers=headers, json=data)
            response.raise_for_status()
            
            export_id = response.json()["id"]
            
            # Poll for completion
            status_url = f"{export_url}({export_id})"
            while True:
                status_response = requests.get(status_url, headers=headers)
                status_response.raise_for_status()
                
                status = status_response.json()["status"]
                if status == "Succeeded":
                    # Download file
                    file_url = f"{status_url}/file"
                    file_response = requests.get(file_url, headers=headers)
                    file_response.raise_for_status()
                    
                    logger.info(f"Report exported successfully as {format}")
                    return file_response.content
                    
                elif status == "Failed":
                    logger.error("Report export failed")
                    return None
                
                time.sleep(5)  # Wait 5 seconds before checking again
                
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return None

def main():
    """Example usage of Power BI integration."""
    pbi = PowerBIIntegration()
    
    # Trigger dataset refresh
    if pbi.refresh_dataset():
        print("✅ Dataset refresh triggered")
        
        # Wait for completion
        if pbi.wait_for_refresh_completion():
            print("✅ Dataset refresh completed successfully")
        else:
            print("❌ Dataset refresh failed or timed out")
    else:
        print("❌ Failed to trigger dataset refresh")
        print("📝 Manual steps required:")
        print("   1. Open Power BI Service")
        print("   2. Navigate to workspace")
        print("   3. Click 'Refresh now' on dataset")

if __name__ == "__main__":
    main()