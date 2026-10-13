from langchain_core.prompts import ChatPromptTemplate

CUSTOMER_FEEDBACK_TEMPLATE = """

You are a customer feedback analyst for an online store.

Analyse the customer feedback provided below.

Extract the following information:

1. sentiment:
    Classify the overall sentiment as positive, neutral, or negative.

2. is_gift:
    Determine whether the product was purchased as a gift.
    Return false if this information is not mentioned.

3. delivery_days:
    Extract how many days delivery took.
    Return -1 if the delivery time is not mentioned.

4. price_assessment:
    Briefly summarise the customer's opinion about the price.
    Return "Not mentioned" if the customer does not discuss price.

Customer feedback:
{customer_feedback}

{format_instructions}
"""

customer_feedback_prompt = ChatPromptTemplate.from_template(CUSTOMER_FEEDBACK_TEMPLATE)
