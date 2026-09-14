from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal

load_dotenv()

# Models
class Details(BaseModel):
    customer_name: str = Field(description="The name of the customer")
    customer_email: str = Field(description="The email of the customer")
    phone_number: str = Field(description="The phone number of the customer")
    delivery_address: str
    order_number: str
    account_number: str
    date: str
    invoice_number: str

# data
email = """
From: sandra.keller@nordwind-logistik.de
To: support@acme-shipping.com
Date: Tue, 8 Sep 2026 09:42:11 +0200
Subject: Order #48213 — wrong item delivered, urgent replacement needed

Hi there,

I placed order #48213 on 28 August through your web shop (account
NL-77219, invoice INV-2026-08-4412). It arrived yesterday at our
Hamburg warehouse, but instead of the 4x "HeavyDuty Pallet Wrap 500m"
we ordered, the box contained 4x "Standard Stretch Film 250m".

We need the correct items before Friday 11 September, otherwise we
can't fulfil a client shipment and will have to source locally at
our own cost.

Could you please:
1. Arrange a replacement delivery to Süderstraße 142, 20537 Hamburg
2. Send a return label for the wrong goods
3. Confirm whether we'll be credited the price difference

This is the second time this has happened this quarter, so I'd also
appreciate someone looking into why the picking keeps going wrong.

You can reach me on +49 40 5566 7788 (weekdays 8–17).

Best regards,
Sandra Keller
Operations Lead, Nordwind Logistik GmbH
"""

# create agent
agent = Agent("openai:gpt-5.4", output_type=Details)

result = agent.run_sync( f"Extract info from: {email}")

print(result.output)
