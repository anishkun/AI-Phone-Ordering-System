import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load the variables from the .env file into the environment
load_dotenv()

# Fetch the variables using os.getenv()
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_NUMBER")
your_mobile_number = os.getenv("MY_MOBILE_NUMBER")
ngrok_url = os.getenv("NGROK_URL")

# Initialize the Twilio Client
client = Client(account_sid, auth_token)

print(f"Calling {your_mobile_number}...")

# Initiate the call
call = client.calls.create(
    to=your_mobile_number,
    from_=twilio_number,
    url=ngrok_url,
    method="POST"
)

print(f"Call initiated! SID: {call.sid}")