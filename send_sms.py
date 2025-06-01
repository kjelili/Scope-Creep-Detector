from twilio.rest import Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def send_sms(to_number, message):
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_PHONE_NUMBER")

        if not all([account_sid, auth_token, from_number]):
            print("Missing Twilio credentials in environment variables.")
            return False

        client = Client(account_sid, auth_token)
        client.messages.create(body=message, from_=from_number, to=to_number)
        print("SMS sent successfully.")
        return True
    except Exception as e:
        print(f"SMS failed: {e}")
        return False
