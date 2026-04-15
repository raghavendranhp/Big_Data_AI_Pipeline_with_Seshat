import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

#load environment variables
load_dotenv()

def generate_anomaly_insight(amount: float, merchant_category_idx: float, location_idx: float, txn_type_idx: float, is_anomaly: bool) -> str:
    """
    Generates a human-readable insight explaining why a transaction might be an anomaly.

    Args:
        amount (float): The transaction amount.
        merchant_category_idx (float): The merchant category index.
        location_idx (float): The location index.
        txn_type_idx (float): The transaction type index.
        is_anomaly (bool): True if the transaction was flagged as an anomaly, False otherwise.

    Returns:
        str: A generated explanation from the language model regarding the transaction.
    """
    #initialize the chatgroq language model
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0
    )

    #define the prompt template for reasoning
    prompt = PromptTemplate(
        input_variables=["amount", "merchant_category_idx", "location_idx", "txn_type_idx", "prediction_status"],
        template=(
            "you are an expert financial fraud analyst. "
            "a transaction has been processed with the following details:\n"
            "amount: {amount}\n"
            "merchant category index: {merchant_category_idx}\n"
            "location index: {location_idx}\n"
            "transaction type index: {txn_type_idx}\n"
            "anomaly prediction: {prediction_status}\n\n"
            "provide a clear, logical, and simple explanation of why this transaction "
            "might be classified as an anomaly or completely normal based on these features. "
            "do not use formatting that includes special visual symbols."
        )
    )
    
    #determine status string
    prediction_status = "anomaly" if is_anomaly else "normal"

    #format the prompt with details
    chain = prompt | llm
    
    #invoke the model and return the result
    result = chain.invoke({
        "amount": amount,
        "merchant_category_idx": merchant_category_idx,
        "location_idx": location_idx,
        "txn_type_idx": txn_type_idx,
        "prediction_status": prediction_status
    })

    return result.content
