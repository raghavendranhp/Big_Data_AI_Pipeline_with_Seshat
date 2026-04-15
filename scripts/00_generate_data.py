import os
import time
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
TOTAL_ROWS = 100_000_000  # 10 Crore
CHUNK_SIZE = 1_000_000    # 10 Lakh per file
NUM_CHUNKS = TOTAL_ROWS // CHUNK_SIZE

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Reference arrays for fast randomized choices
MERCHANT_CATEGORIES = ['Retail', 'Travel', 'Food', 'Electronics', 'Digital_Services']
LOCATIONS = ['Domestic', 'International']
TXN_TYPES = ['POS', 'Online', 'ATM']

print(f"Starting generation of {TOTAL_ROWS:,} rows in {NUM_CHUNKS} chunks...")
print(f"Destination: {RAW_DATA_DIR}\n")

start_time = time.time()

# ==========================================
# GENERATION LOOP
# ==========================================
for chunk_id in range(1, NUM_CHUNKS + 1):
    chunk_start_time = time.time()
    
    # 1. Generate core numerical/categorical arrays using NumPy (Extremely Fast)
    transaction_ids = np.arange((chunk_id - 1) * CHUNK_SIZE, chunk_id * CHUNK_SIZE)
    user_ids = np.random.randint(1000, 500000, size=CHUNK_SIZE)
    
    # Base transaction amounts (Log-normal distribution simulating real transactions)
    # Most txns are small (₹500 - ₹5000), some are very large
    amounts = np.round(np.random.lognormal(mean=7.0, sigma=1.5, size=CHUNK_SIZE), 2)
    
    categories = np.random.choice(MERCHANT_CATEGORIES, size=CHUNK_SIZE)
    locations = np.random.choice(LOCATIONS, size=CHUNK_SIZE, p=[0.85, 0.15])
    txn_types = np.random.choice(TXN_TYPES, size=CHUNK_SIZE)
    
    # 2. Inject Anomaly / Fraud Logic (Seshat Pattern Target)
    # Pattern: International ATM or Online transactions over 15,000 have a high fraud risk
    is_anomaly = np.zeros(CHUNK_SIZE, dtype=int)
    
    # Condition masks
    high_value_mask = amounts > 15000
    risk_loc_mask = locations == 'International'
    risk_type_mask = np.isin(txn_types, ['Online', 'ATM'])
    
    # Apply logic: 80% chance of being marked anomaly if conditions met
    combined_mask = high_value_mask & risk_loc_mask & risk_type_mask
    anomaly_chances = np.random.random(size=CHUNK_SIZE)
    is_anomaly[combined_mask & (anomaly_chances > 0.2)] = 1
    
    # Add random noise (0.1% random anomalies for noise)
    noise_mask = np.random.random(size=CHUNK_SIZE) < 0.001
    is_anomaly[noise_mask] = 1

    # 3. Assemble into Pandas DataFrame and save
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'user_id': user_ids,
        'amount': amounts,
        'merchant_category': categories,
        'location': locations,
        'txn_type': txn_types,
        'is_anomaly': is_anomaly
    })
    
    # Introduce ~2% missing values in 'merchant_category' to satisfy Day 1 cleaning requirements
    missing_mask = np.random.random(size=CHUNK_SIZE) < 0.02
    df.loc[missing_mask, 'merchant_category'] = np.nan
    
    # Save to partitioned CSV
    file_path = os.path.join(RAW_DATA_DIR, f"transactions_part_{chunk_id:03d}.csv")
    df.to_csv(file_path, index=False)
    
    chunk_time = time.time() - chunk_start_time
    print(f"Created Chunk {chunk_id:03d}/{NUM_CHUNKS} -> {file_path} (Took {chunk_time:.2f}s)")

total_time = time.time() - start_time
print(f"\n✅ Data generation complete! 10 Crore rows generated.")
print(f"Total execution time: {total_time / 60:.2f} minutes")