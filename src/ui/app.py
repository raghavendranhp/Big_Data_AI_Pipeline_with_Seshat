import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler
import os
import sys

#add src directory to path to import reasoning engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.llm.reasoning_engine import generate_anomaly_insight

def load_spark_model():
    """
    Loads the saved PySpark Random Forest anomaly detection model.

    Returns:
        The loaded PySpark machine learning model (either PipelineModel or RandomForestClassificationModel).
    """
    #initialize spark session
    spark = SparkSession.builder \
        .appName("seshat_anomaly_detection") \
        .getOrCreate()
        
    #define model path
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "seshat_anomaly_model"))
    
    #load and return the model
    #attempt to load as pipeline model first
    try:
        model = PipelineModel.load(model_path)
    except Exception:
        #fallback to random forest model
        model = RandomForestClassificationModel.load(model_path)
    return model

def create_ui() -> None:
    """
    Constructs the Streamlit user interface, handling the navigation menu
    and rendering the selected tabs for visualization, detection, and reasoning.

    Returns:
        None
    """
    #configure the page settings
    st.set_page_config(page_title="big data pipeline and intelligent insights", layout="wide")
    
    #create horizontal navigation menu
    selected_tab = option_menu(
        menu_title=None,
        options=["visualization", "dynamic detection"],
        icons=["", ""],
        menu_icon=None,
        default_index=0,
        orientation="horizontal",
    )
    
    if selected_tab == "visualization":
        render_visualization_tab()
    elif selected_tab == "dynamic detection":
        render_detection_tab()

def render_visualization_tab() -> None:
    """
    Renders the visualization tab, showing basic historical metrics
    and placeholder charts.

    Returns:
        None
    """
    st.title("visualization dashboard")
    st.write("historical metrics and transaction data overview.")
    
    #create placeholder dataframe for basic chart
    data = {"date": ["jan", "feb", "mar", "apr", "may"], "anomalies": [10, 15, 8, 20, 12]}
    df = pd.DataFrame(data)
    
    #display line chart of anomalies
    st.line_chart(df, x="date", y="anomalies")
    
    #display metrics
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="total transactions", value="ten thousand")
    with col_b:
        st.metric(label="detected anomalies", value="sixty five")

def render_detection_tab() -> None:
    """
    Renders the dynamic detection tab, providing form inputs to accept
    new transaction data, passing it to the PySpark model, and displaying
    the detection result.

    Returns:
        None
    """
    st.title("dynamic transaction detection")
    st.write("enter transaction details to test against the trained random forest model.")
    
    #create input form
    with st.form("transaction_form"):
        amount = st.number_input("transaction amount", min_value=0.0, value=100.0)
        merchant_category_idx = st.number_input("merchant category index", min_value=0.0, value=1.0)
        location_idx = st.number_input("location index", min_value=0.0, value=1.0)
        txn_type_idx = st.number_input("transaction type index", min_value=0.0, value=1.0)
        
        submit_button = st.form_submit_button(label="predict anomaly")
        
    if submit_button:
        #load spark session and model
        try:
            spark = SparkSession.builder.getOrCreate()
            model = load_spark_model()
            
            #create dataframe for the single input row
            #model expects the specific columns used during training
            input_data = [(float(amount), float(merchant_category_idx), float(location_idx), float(txn_type_idx))]
            df = spark.createDataFrame(input_data, ["amount", "merchant_category_idx", "location_idx", "txn_type_idx"])
            
            #check if model requires vector assembly
            if isinstance(model, RandomForestClassificationModel):
                #assemble features manually
                assembler = VectorAssembler(
                    inputCols=["amount", "merchant_category_idx", "location_idx", "txn_type_idx"],
                    outputCol="features"
                )
                df_to_transform = assembler.transform(df)
            else:
                #pipeline model handles assembly
                df_to_transform = df
            
            #run prediction
            predictions = model.transform(df_to_transform)
            
            #extract result
            #prediction column is outputted by transform
            result_row = predictions.select("prediction").collect()[0]
            prediction_val = result_row["prediction"]
            
            #store in session state for reasoning tab
            st.session_state["last_transaction"] = {
                "amount": amount,
                "merchant_category_idx": merchant_category_idx,
                "location_idx": location_idx,
                "txn_type_idx": txn_type_idx,
                "is_anomaly": bool(prediction_val > 0.0)
            }
            
            if prediction_val > 0.0:
                st.error("alert: transaction flagged as an anomaly.")
            else:
                st.success("transaction appears normal.")
                
        except Exception as e:
            st.error(f"model evaluation failed: {str(e)}")
            
    #render reasoning section below the prediction form
    st.divider()
    render_reasoning()

def render_reasoning() -> None:
    """
    Renders the seshat reasoning section directly beneath the prediction,
    providing a button to trigger langchain and groq for generating a 
    logical explanation.

    Returns:
        None
    """
    st.title("seshat reasoning engine")
    st.write("generate automated explanations for the most recent transaction prediction.")
    
    if "last_transaction" not in st.session_state:
        st.warning("please process a transaction in the dynamic detection tab first.")
        return
        
    #extract transaction details from session state
    txn_data = st.session_state["last_transaction"]
    
    st.write("current transaction context:")
    st.json(txn_data)
    
    if st.button("generate logical explanation"):
        with st.spinner("consulting the large language model..."):
            try:
                #call reasoning engine to generate explanation
                explanation = generate_anomaly_insight(
                    amount=txn_data["amount"],
                    merchant_category_idx=txn_data["merchant_category_idx"],
                    location_idx=txn_data["location_idx"],
                    txn_type_idx=txn_data["txn_type_idx"],
                    is_anomaly=txn_data["is_anomaly"]
                )
                
                st.write("### generated insight")
                st.write(explanation)
            except Exception as e:
                st.error(f"failed to generate insight: {str(e)}")

if __name__ == "__main__":
    #run the main ui function
    create_ui()
