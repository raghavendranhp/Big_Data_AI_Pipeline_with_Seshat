# Big Data Pipeline & Intelligent Insights (Seshat AI)

## Overview
This project implements a scalable, end-to-end Big Data pipeline designed to process large-scale financial transaction data, detect fraud anomalies, and generate intelligent, human-readable insights. The architecture is inspired by Seshat AI concepts, focusing on knowledge extraction, pattern discovery, and rule-based reasoning.

## Tech Stack
* **Big Data Engine:** Apache Spark (PySpark), Spark SQL
* **Machine Learning:** Spark MLlib (Distributed Random Forest)
* **AI & Reasoning:** LangChain, Groq API (LLM Integration)
* **Frontend UI:** Streamlit, Streamlit-Option-Menu
* **Data Processing:** Pandas, NumPy
* **Storage:** Parquet (Simulated HDFS Data Lake)

## Pipeline Architecture

### Phase 1: Ingestion & Preprocessing
* Ingests millions of rows of synthetic transaction data.
* Performs automated data cleaning and handles missing values.
* Engineers features using PySpark `StringIndexer` and `VectorAssembler`.
* Saves highly compressed data into a Parquet-based data lake.

### Phase 2: Distributed Machine Learning
* Trains a distributed Random Forest classification model to detect anomalies.
* Extracts "knowledge rules" by calculating mathematical feature importances (e.g., Location and Amount driving fraud risk).
* Serializes and saves the model artifacts for low-latency downstream inference.

### Phase 3: Insight Engine & Visualization
* Provides a dynamic Streamlit dashboard for real-time interaction.
* **Dynamic Detection:** Accepts new transaction inputs and passes them through the serialized Spark model for instant risk prediction.
* **Seshat Reasoning:** Utilizes LangChain and an LLM to translate numerical model outputs and feature importances into clear, human-readable explanations of why an anomaly was flagged.

## Directory Structure
```text
seshat_bigdata_project/
├── data/
│   ├── raw/                              
│   └── processed/                        
├── models/
│   └── seshat_anomaly_model/             
├── notebooks/
│   ├── 01_data_ingestion_preprocessing.ipynb
│   └── 02_seshat_ml_modeling.ipynb       
├── scripts/
│   └── 00_generate_data.py               
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py                    
│   │   └── reasoning_engine.py           
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py                        
│   │   └── components.py                 
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                    
├── .env
├── requirements.txt                      
└── README.md
```

## Setup & Installation

**1. Create and Activate Virtual Environment**
```bash
python -m venv .venv

#for windows
.venv\Scripts\activate

#for linux/mac
source .venv/bin/activate
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Environment Variables**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

## Execution Guide

**Step 1: Generate the Data Lake**
Run the data generation script to create the synthetic transaction data.
```bash
python scripts/00_generate_data.py
```

**Step 2: Big Data Processing**
Execute `notebooks/01_data_ingestion_preprocessing.ipynb` to clean the data and convert it to Parquet format.

**Step 3: Model Training**
Execute `notebooks/02_seshat_ml_modeling.ipynb` to train the distributed Random Forest model and extract the Seshat knowledge base.

**Step 4: Launch the Insight Engine**
Start the Streamlit dashboard to interact with the model and AI reasoning engine.
```bash
streamlit run src/ui/app.py
```
