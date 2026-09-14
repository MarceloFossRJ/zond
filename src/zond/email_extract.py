from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

load_dotenv()

# create llm
llm = ChatOpenAI(model="gpt-5.4", temperature=0)

class details(BaseModel):
    customer_name: str
    customer_email: str
    phone_number: str
    delivery_address: str
    order_number: str
    account_number: str
    date: str
    invoice_number: str

details_schema = details.model_json_schema()

vars = " customer_name, customer_email, phone_number, delivery_address, order_number, account_number, date, invoice_number"

# data
context = """
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


question = f"Please extract {vars} from the provided email in the context"

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
 Use only the provided context to answer the question.
 If the answer is not in the context, say: "I could not find in the document" 
 Context:
 {context}
 Question:
 {question}
 """)

# build a chain
chain = prompt | llm | StrOutputParser()

# get response
response = chain.invoke({
    "context": context,
    "question": question
})

print("\n Answer: ")
print(response)
print("\n" + "-" * 20 + "\n")