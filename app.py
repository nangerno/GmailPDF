from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import base64
import openai
import PyPDF2
import io

load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')
SECRET_KEY = os.getenv('SECRET_KEY')

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

oauth = OAuth()
oauth.register(
    name='google',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile https://www.googleapis.com/auth/gmail.readonly',
        'access_type': 'offline',
        'prompt': 'consent'
    }
)
openai.api_key = OPENAI_APIKEY

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = request.session.get('user')
    if user:
        return templates.TemplateResponse("home.html", {"request": request, "user": user})
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/contactus", response_class=HTMLResponse)
async def contactus(request: Request):
    return templates.TemplateResponse("contactus.html", {"request": request})

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user = token.get('userinfo')
        if not user:
            user = await oauth.google.userinfo(token=token)
        request.session['user'] = dict(user)
        request.session['token'] = dict(token)
        return RedirectResponse(url='/')
    except OAuthError as error:
        return HTMLResponse(f"<h1>{error.error}</h1>")

@app.get("/logout")
async def logout(request: Request):
    request.session.pop('user', None)
    request.session.pop('token', None)
    return RedirectResponse(url='/')

@app.post("/download_pdfs")
async def download_pdfs(request: Request):
    # Get the selected PDF ids from the form data
    form_data = await request.form()
    pdf_ids = form_data.getlist("pdf_ids")

    # Fetch the user's OAuth token from the session
    token = request.session.get('token')
    if not token:
        raise HTTPException(status_code=401, detail="User not authenticated")
    
    credentials = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Build the Gmail service
    gmail_service = build('gmail', 'v1', credentials=credentials)

    # Prepare the PDF data for downloading
    pdf_data_list = []

    for pdf_id in pdf_ids:
        # Fetch the email message
        msg = gmail_service.users().messages().get(userId='me', id=pdf_id).execute()
        subject = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject')
        attachments = msg['payload'].get('parts', [])
        
        for attachment in attachments:
            if 'filename' in attachment and attachment['filename'].endswith('.pdf'):
                attachment_id = attachment['body']['attachmentId']
                attachment_data = gmail_service.users().messages().attachments().get(userId='me', messageId=pdf_id, id=attachment_id).execute()
                data = attachment_data['data']
                pdf_data = base64.urlsafe_b64decode(data.encode('UTF-8'))

                # Store the PDF data for later individual download
                pdf_data_list.append({
                    'filename': attachment['filename'],
                    'data': pdf_data
                })
    
    if not pdf_data_list:
        raise HTTPException(status_code=404, detail="No PDF attachments found.")
    
    # Now we will return each PDF as an individual response
    # For each PDF in the list, we will return a separate StreamingResponse
    responses = []
    for pdf in pdf_data_list:
        pdf_filename = pdf['filename']
        pdf_data = pdf['data']

        # Create a StreamingResponse for each PDF
        pdf_response = StreamingResponse(
            io.BytesIO(pdf_data), 
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename={pdf_filename}"}
        )
        responses.append(pdf_response)
    
    # Return the first response for simplicity; however, you can adjust this depending on the client-side handling
    # of multiple files.
    return responses[0]
@app.get("/search")
async def search(request: Request, query: str = None):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required.")
    
    # Fetch the user's OAuth token from the session
    token = request.session.get('token')
    credentials = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Build the Gmail service
    gmail_service = build('gmail', 'v1', credentials=credentials)

    # Modify the search query to find PDFs matching the user input keyword
    search_query = f"has:attachment filename:pdf {query}"
    results = gmail_service.users().messages().list(userId='me', q=search_query).execute()
    messages = results.get('messages', [])

    pdf_list = []
    for message in messages:
        msg = gmail_service.users().messages().get(userId='me', id=message['id']).execute()
        subject = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject')
        attachments = msg['payload'].get('parts', [])
        
        for attachment in attachments:
            if 'filename' in attachment and attachment['filename'].endswith('.pdf'):
                attachment_id = attachment['body']['attachmentId']
                attachment_data = gmail_service.users().messages().attachments().get(userId='me', messageId=message['id'], id=attachment_id).execute()
                data = attachment_data['data']
                pdf_data = base64.urlsafe_b64decode(data.encode('UTF-8'))

                # Extract the first 100 sentences of the PDF
                pdf_text_chunks = extract_pdf_first_100_sentences(pdf_data)
                
                # Analyze the PDF with GPT based on the user's search query
                gpt_response = analyze_pdf_with_gpt(pdf_text_chunks, query)
                
                pdf_list.append({
                    "id": message['id'],
                    "subject": subject,
                    "attachment": attachment['filename'],
                    "gpt_summary": gpt_response
                })

    return templates.TemplateResponse("results.html", {"request": request, "pdfs": pdf_list})

def extract_pdf_first_100_sentences(pdf_data: bytes) -> str:
    """
    Extract text from a PDF and retrieve the first 100 sentences.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Extract first 100 sentences
        sentences = text.split('.')
        return '. '.join(sentences[:100])
    except PyPDF2.errors.DependencyError as e:
        print(f"Decryption Error: {e}")
        return "Error decrypting PDF."
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return "Error processing PDF."

def analyze_pdf_with_gpt(pdf_content: str, query: str) -> str:
    """
    Analyze PDF content with GPT and return a summarized result.
    """
    try:
        prompt = (
            f"Analyze the content below to classify it according to the query: '{query}'\n\n"
            f"Content:\n{pdf_content}\n"
            "Provide a concise summary and classification of the relevant information."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[ 
                {"role": "system", "content": "You are a professional assistant with expertise in text analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error with GPT analysis: {e}")
        return "Error analyzing PDF content."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
