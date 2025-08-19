from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_service():
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)

    creds = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=creds)

def fetch_emails(service, max_results=10):
    result = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = result.get('messages', [])
    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        snippet = txt.get('snippet', '')
        emails.append(snippet)
    return emails
