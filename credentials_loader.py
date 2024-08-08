import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2 import service_account
from dotenv import load_dotenv

class CredentialsLoader:
    SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

    @staticmethod
    def load_creds():
        creds = None
        if os.path.exists('env/token.json'):
            creds = Credentials.from_authorized_user_file('env/token.json', CredentialsLoader.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('env/client_secret.json', CredentialsLoader.SCOPES)
                creds = flow.run_local_server(port=0)
            with open('env/token.json', 'w') as token:
                token.write(creds.to_json())
        return creds

    @staticmethod
    def load_iam_creds():
        creds = service_account.Credentials.from_service_account_file('env/gen-lang-client.json')
        return creds
