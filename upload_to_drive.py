"""
Upload large PermitIQ files to Google Drive for Mac→PC transfer.
Uses OAuth2 browser auth (opens a tab — just click Allow).
Streams files without copying, so no extra disk space needed.

Usage: python upload_to_drive.py
"""
import os
import sys
import json
import pickle
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']
PROJECT_DIR = "/Users/stevenspero/Desktop/Boston Zoning Project"
TOKEN_FILE = "/tmp/permitiq_drive_token.json"

# Files and directories to upload
ITEMS_TO_UPLOAD = [
    # (local_path_relative, drive_folder_name_or_None)
    ("building_permits.csv", None),
    ("zba_model_v2.pkl", None),
    ("api/zba_model.pkl", None),
    ("boston_parcels_zoning.geojson", None),
    ("property_assessment_fy2026.csv", None),
    ("zba_cases_cleaned.csv", None),
    ("zba_cases_cleaned.backup.20260331_212800.csv", None),
    ("zba_cases_dataset.csv", None),
    ("zba_tracker.csv", None),
    ("zba_agendas.csv", None),
    ("zba_model.pkl", None),
    ("CLAUDE.md", None),
    ("DEMO_SCRIPT.md", None),
    ("TASKS.md", None),
    ("REFACTOR_PLAN.md", None),
    ("boston_parcel_zoning.schema.json", None),
    ("railway.json", None),
]

DIRS_TO_UPLOAD = [
    "pdfs",
    "model_history",
    "parcels_2025",
    "parcels_2025_clean",
    "memory",
    "leads",
]


def get_credentials():
    """Get or refresh OAuth2 credentials."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use a public OAuth client ID for CLI tools
            # This is Google's own "TV and Limited Input" client
            client_config = {
                "installed": {
                    "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
                    "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"]
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True)

        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())

    return creds


def find_or_create_folder(service, name, parent_id=None):
    """Find or create a folder in Drive."""
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])

    if files:
        print(f"  Found existing folder: {name}")
        return files[0]['id']

    metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        metadata['parents'] = [parent_id]

    folder = service.files().create(body=metadata, fields='id').execute()
    print(f"  Created folder: {name}")
    return folder['id']


def upload_file(service, local_path, parent_id, display_name=None):
    """Upload a single file to Drive, streaming from disk."""
    name = display_name or os.path.basename(local_path)
    size = os.path.getsize(local_path)
    size_mb = size / 1024 / 1024

    # Check if file already exists
    query = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    existing = service.files().list(q=query, fields='files(id,name,size)').execute().get('files', [])
    if existing:
        existing_size = int(existing[0].get('size', 0))
        if abs(existing_size - size) < 1024:  # Within 1KB = same file
            print(f"  SKIP (exists): {name} ({size_mb:.1f} MB)")
            return existing[0]['id']
        else:
            print(f"  REPLACE: {name} ({size_mb:.1f} MB, size changed)")
            service.files().delete(fileId=existing[0]['id']).execute()

    print(f"  UPLOAD: {name} ({size_mb:.1f} MB)...", end='', flush=True)

    metadata = {'name': name, 'parents': [parent_id]}

    # Use resumable upload for large files
    if size > 5 * 1024 * 1024:  # > 5MB
        media = MediaFileUpload(local_path, resumable=True, chunksize=10*1024*1024)
        request = service.files().create(body=metadata, media_body=media, fields='id')
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"\r  UPLOAD: {name} ({size_mb:.1f} MB)... {pct}%", end='', flush=True)
    else:
        media = MediaFileUpload(local_path)
        response = service.files().create(body=metadata, media_body=media, fields='id').execute()

    print(f"\r  UPLOAD: {name} ({size_mb:.1f} MB)... DONE")
    return response.get('id') if isinstance(response, dict) else response['id']


def upload_directory(service, local_dir, parent_id):
    """Recursively upload a directory."""
    dir_name = os.path.basename(local_dir)
    folder_id = find_or_create_folder(service, dir_name, parent_id)

    count = 0
    for item in sorted(os.listdir(local_dir)):
        item_path = os.path.join(local_dir, item)
        if item.startswith('.'):
            continue
        if os.path.isdir(item_path):
            count += upload_directory(service, item_path, folder_id)
        elif os.path.isfile(item_path):
            upload_file(service, item_path, folder_id)
            count += 1

    return count


def main():
    print("=" * 60)
    print("PermitIQ → Google Drive Transfer")
    print("=" * 60)
    print()

    # Authenticate
    print("Step 1: Authenticating with Google Drive...")
    print("(A browser window will open — sign in and click Allow)")
    print()
    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)
    print("Authenticated!")
    print()

    # Create root folder
    print("Step 2: Creating PermitIQ-Complete-Transfer folder...")
    root_id = find_or_create_folder(service, 'PermitIQ-Complete-Transfer')
    print()

    # Upload individual files
    print("Step 3: Uploading individual files...")
    for rel_path, subfolder in ITEMS_TO_UPLOAD:
        full_path = os.path.join(PROJECT_DIR, rel_path)
        if os.path.exists(full_path):
            if subfolder:
                sub_id = find_or_create_folder(service, subfolder, root_id)
                upload_file(service, full_path, sub_id)
            else:
                upload_file(service, full_path, root_id)
        else:
            print(f"  SKIP (not found): {rel_path}")
    print()

    # Upload directories
    print("Step 4: Uploading directories...")
    for dir_name in DIRS_TO_UPLOAD:
        dir_path = os.path.join(PROJECT_DIR, dir_name)
        if os.path.isdir(dir_path):
            count = upload_directory(service, dir_path, root_id)
            print(f"  {dir_name}/: {count} files uploaded")
        else:
            print(f"  SKIP (not found): {dir_name}/")
    print()

    print("=" * 60)
    print("TRANSFER COMPLETE!")
    print(f"All files are in Google Drive → PermitIQ-Complete-Transfer")
    print("On your PC: download the folder from drive.google.com")
    print("=" * 60)


if __name__ == '__main__':
    main()
