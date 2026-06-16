# Learn what Pydantic is, what it’s used for, 
# and how it compares to alternatives

def calculate_user_discount(age: int, is_premium_member: bool, purchase_amount: float) -> float:
   """Calculate discount percentage based on user profile and purchase amount."""
   if age >= 65:
       base_discount = 0.15
   elif is_premium_member:
       base_discount = 0.10
   else:
       base_discount = 0.05
  
   return purchase_amount * base_discount

# This runs without error, even though types are completely wrong!
discount = calculate_user_discount(True, 1, 5)
print(discount)  # Output: 0.5 (True becomes 1, 1 is truthy, so 5 * 0.10)

'''
This is a classic example of Python’s dynamic typing. It gives programmers fast development and freedom, 
but this comes at the cost of introducing type validation issues that often surface in production.
'''

'''
Pydantic takes untrusted data, checks it while the program is running, 
converts it when appropriate, and either gives you a dependable Python object or raises a clear error.

Outside world                    Your application
---------------------------------------------------------
API request       ───────┐
Database result   ───────┤
CSV row           ───────┤──> PYDANTIC GATE ──> Trusted objects
LLM response      ───────┤
Environment vars  ───────┤
User input        ───────┘

Almost every Pydantic implementation follows these four steps:

1. Describe the expected data
class Something(BaseModel):
    ...

2. Receive raw data
raw_data = {...}

3. Validate it
something = Something.model_validate(raw_data)

4. Use the trusted object
print(something.some_field)

** Remember this sentence: Define → receive → validate → use.
'''


from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class User(BaseModel):
   age: int
   email: EmailStr
   is_active: bool = True
   nickname: Optional[str] = None

# Pydantic automatically validates and converts data
user_data = {
   "age": "25",  # String gets converted to int
   "email": "john@example.com",
   "is_active": "true"  # String gets converted to bool
}

user = User.model_validate(user_data)
print(user)
print(user.age)  # 25 (as integer)
print(type(user.age))  # 25 (as integer)
print(user.model_dump())  # Clean dictionary output


# Pydantic is excellent for nested data 
# Real applications rarely have completely flat data.

class Customer(BaseModel):
    name: str
    email: str

class OrderItem(BaseModel):
    product_name: str
    price: float = Field(gt=0)
    quantity: int = Field(gt=0)

class Order(BaseModel):
    order_id: int
    customer: Customer
    items: list[OrderItem]


raw_order = {
    "order_id": 501,
    "customer": {
        "name": "Tanvi",
        "email": "tanvi@example.com"
    },
    "items": [
        {
            "product_name": "Keyboard",
            "price": 50,
            "quantity": 1
        },
        {
            "product_name": "Mouse",
            "price": 25,
            "quantity": 2
        }
    ]
}

order = Order.model_validate(raw_order)

print(order.customer.name)
print(order.items[0].product_name)
print(order.items[1].quantity)

# Enums: restrict a value to known choices

'''
Suppose an order status can only be:

pending
shipped
delivered
cancelled
'''

from enum import Enum #  Enum (enumeration) is a class used to define a finite set of named constants (members)

class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Order(BaseModel):
    order_id: int
    status: OrderStatus

order = Order(
    order_id=101,
    status="pending"
)

print(order.status) # will give error for anything outside the allowed statusses in order status class


# strict validation in pydantic
# Use strict mode when automatic conversion could hide a bug
from pydantic import BaseModel, Field

class Person(BaseModel):
    age: int = Field(strict=True)

print(Person(age=25))
# print(Person(age='25')) # THIS WILL GIVE ERROR


# Reject unknown fields
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(extra="forbid") # will help reject unkown fields and maintain consistency
    name: str
    age: int

# print(                     # WILL THROW ERROR
#     User(
#     name="Tanvi",
#     age=25,
#     admin=True
# )
# )


# Turning a model back into data
# Pydantic does two major jobs:

'''
Raw data → Pydantic object     Validation
Pydantic object → dict/JSON    Serialization
'''

# convert to dictionary
user_dict = user.model_dump()
print(user_dict)

# convert to json
user_json = user.model_dump_json()
print(user_json)

############################################################
'''
Pydantic in AI systems
Pydantic is especially valuable in AI because language-model output is otherwise just text.

Imagine asking an LLM: Classify this support ticket.
It might produce: This appears to be a billing issue with fairly high urgency.

our application cannot reliably program against that sentence.

You want:

{
    "category": "billing",
    "urgency": "high",
    "confidence": 0.91
}
'''
############################################################

from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# define the contract
class TicketClassification(BaseModel):
    category: Literal["billing","technical","account","other"]
    urgency: Literal["low","medium","high"]
    confidence: float = Field(ge=0,le=1)
    explanation: str

# Now the AI output has a known shape
client = OpenAI()
response = client.responses.parse(
    model="gpt-5.5",
    input="Write a short bedtime story about a unicorn.",
    text_format=TicketClassification
)

print(response.output_text)
print()
print('############################### Pydantic Assignment ############################')

'''
Build a small agentic AI system where the LLM cannot freely return random text. Every important decision must be validated with Pydantic.
Create an AI assistant that receives a user request and decides what kind of agent workflow to run.


Example req:
"Summarize this resume"
"Find missing skills in this candidate profile"
"Generate interview questions for this candidate"
"Store this as memory: I prefer short answers"
"Compare this candidate with a job description"

Rules

Do not trust raw LLM text.
Every important LLM output must be validated with Pydantic.
The route should only allow fixed choices.
Confidence should be a number between 0 and 1.
Memory should only be stored if the request actually contains useful user preference or personal context.
Unsupported requests should not be forced into a resume-related route.
'''