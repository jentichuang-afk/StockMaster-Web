"""
google_drive.py - Google Drive 觀察清單同步工具

使用 OAuth 2.0 授權後，從使用者的 Google Drive 讀取/寫入
名為 'stock_master_watchlist.txt' 的觀察清單檔案。
"""

import json
import io
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import streamlit as st

# Google Drive 的存取範圍（只存取本 App 建立的檔案）
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]

WATCHLIST_FILENAME = "stock_master_watchlist.txt"
REDIRECT_URI = "http://localhost:8501"


def _get_client_config():
    """從 st.secrets 讀取 OAuth 設定，建立 client_config dict"""
    client_id = st.secrets.get("GOOGLE_CLIENT_ID", "")
    client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }


import os
import tempfile
import json

# 允許實際拿到的權限跟請求的不一樣，避免 oauthlib 拋出異常
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

STATE_FILE = os.path.join(tempfile.gettempdir(), "stockmaster_oauth_state.json")

def _save_code_verifier(state, verifier):
    data = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
        except:
            pass
    data[state] = verifier
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f)

def _get_code_verifier(state):
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                return data.get(state)
        except:
            pass
    return None

def get_google_auth_url():
    """產生 Google OAuth 授權網址，供使用者點擊登入"""
    config = _get_client_config()
    if not config:
        return None, "請先在 secrets.toml 中設定 GOOGLE_CLIENT_ID 與 GOOGLE_CLIENT_SECRET"

    flow = Flow.from_client_config(
        config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    # 設定 access_type=offline 以取得 refresh_token
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = state
    
    # 將 PKCE verifier 存入暫存檔，避免 Streamlit 重新載入時遺失
    if hasattr(flow, "code_verifier"):
        _save_code_verifier(state, flow.code_verifier)
    
    return auth_url, None


def handle_oauth_callback(code: str, state: str = ""):
    """
    接收 OAuth 回呼的 code，交換 token 並儲存到 session_state。
    成功回傳 True，失敗回傳 False。
    """
    config = _get_client_config()
    if not config:
        return False

    try:
        flow = Flow.from_client_config(
            config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI,
            state=state,
        )
        
        # 嘗試從暫存檔恢復 PKCE verifier
        verifier = _get_code_verifier(state)
        if verifier:
            flow.code_verifier = verifier
            
        flow.fetch_token(code=code)
        creds = flow.credentials
        
        # 檢查是否有給予 Google Drive 的權限
        granted_scopes = list(creds.scopes) if creds.scopes else []
        if "https://www.googleapis.com/auth/drive.file" not in granted_scopes and creds.scopes is not None:
             st.session_state["google_auth_error"] = "您沒有勾選 Google Drive 存取權限。為了能儲存觀察清單，請重新登入並在授權畫面上「勾選」Google Drive 的選項。"
             return False

        # 將 credentials 序列化存入 session_state
        st.session_state["google_creds"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": granted_scopes,
        }
        # 取得使用者 Email
        service = build("oauth2", "v2", credentials=creds)
        user_info = service.userinfo().get().execute()
        st.session_state["google_user_email"] = user_info.get("email", "")
        return True
    except Exception as e:
        st.session_state["google_auth_error"] = str(e)
        return False


def _get_credentials():
    """從 session_state 重建 Credentials 物件"""
    creds_data = st.session_state.get("google_creds")
    if not creds_data:
        return None
    return Credentials(
        token=creds_data["token"],
        refresh_token=creds_data.get("refresh_token"),
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data["scopes"],
    )


def _get_drive_service():
    """建立 Google Drive API 服務物件"""
    creds = _get_credentials()
    if not creds:
        return None
    return build("drive", "v3", credentials=creds)


def _find_watchlist_file(service):
    """在 Drive 中搜尋觀察清單檔案，回傳 file_id 或 None"""
    query = f"name='{WATCHLIST_FILENAME}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def load_from_drive():
    """
    從 Google Drive 讀取觀察清單字串。
    成功回傳字串，失敗回傳 None。
    """
    service = _get_drive_service()
    if not service:
        return None

    try:
        file_id = _find_watchlist_file(service)
        if not file_id:
            return None  # 檔案不存在，代表是新使用者

        request = service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buffer.seek(0)
        return buffer.read().decode("utf-8").strip()
    except Exception as e:
        st.warning(f"⚠️ 從 Google Drive 讀取失敗：{e}")
        return None


def save_to_drive(tickers: str):
    """
    將觀察清單字串寫入 Google Drive。
    成功回傳 True，失敗回傳 False。
    """
    service = _get_drive_service()
    if not service:
        return False

    try:
        file_metadata = {"name": WATCHLIST_FILENAME, "mimeType": "text/plain"}
        content = tickers.encode("utf-8")
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype="text/plain")

        file_id = _find_watchlist_file(service)
        if file_id:
            # 更新現有檔案
            service.files().update(
                fileId=file_id, media_body=media
            ).execute()
        else:
            # 建立新檔案
            service.files().create(
                body=file_metadata, media_body=media, fields="id"
            ).execute()
        return True
    except Exception as e:
        st.warning(f"⚠️ 寫入 Google Drive 失敗：{e}")
        return False


def is_logged_in():
    """檢查目前是否已登入 Google"""
    return "google_creds" in st.session_state


def logout():
    """清除所有 Google 登入資訊"""
    for key in ["google_creds", "google_user_email", "oauth_state", "google_auth_error"]:
        st.session_state.pop(key, None)
