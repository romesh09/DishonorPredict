import os
import json
import numpy as np
import pandas as pd
import gspread
import onnxruntime as rt
from fastapi import FastAPI, HTTPException, BackgroundTasks
from google.oauth2.service_account import Credentials

!pip install onnxruntime # Install onnxruntime for ONNX model inference

app = FastAPI(title="RA16 ONNX Live Inference Service")

# SPREADSHEET CONFIGURATION
PREDICTION_SPREADSHEET_ID = "18nbA_0B0AHpuAYus-zZbKj_HSYAYpbVTdC83lA_D9ZQ"
PREDICTION_TAB_NAME = "dishonorPredict"
OPERATIONAL_THRESHOLD = 0.10

# STANDARDIZED FILE PATHS MATCHING REPO ARCHITECTURE
# Corrected: __file__ is not defined in an interactive environment, use absolute path based on repository location.
ARTIFACTS_DIR = os.path.abspath(os.path.join("/content/DishonorPredict", "prediction service", "artifacts"))
ONNX_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rf_dishonor_model.onnx")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

def get_gspread_client():
    vault_json_string = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
    if not vault_json_string:
        raise HTTPException(status_code=500, detail="Missing GCP_SERVICE_ACCOUNT_JSON variable.")
    credentials_dict = json.loads(vault_json_string)
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    return gspread.authorize(Credentials.from_service_account_info(credentials_dict, scopes=scopes))

def run_inference_pipeline():
    # Structural verification check for model assets
    if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(METADATA_PATH):
        print(f"❌ Error: Artifact files missing at {ARTIFACTS_DIR}")
        return

    # 1. Load ONNX Session & Training Metadata mapping
    session = rt.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    overall_avg_risk = metadata["overall_avg_risk"]
    agent_risk = metadata["agent_risk"]

    # 2. Get Live Production Rows from Google Sheet
    gc = get_gspread_client()
    workbook_prediction = gc.open_by_key(PREDICTION_SPREADSHEET_ID)
    production_sheet = workbook_prediction.worksheet(PREDICTION_TAB_NAME)
    production_df = pd.DataFrame(production_sheet.get_all_records())

    if len(production_df) == 0:
        print("⚠️ Sheet is empty. No active rows to process today.")
        return

    # Clean headers and drop legacy prediction columns if present
    production_df.columns = production_df.columns.str.strip()
    existing_prediction_cols = ['Dishonor Probability', 'ML prediction']
    production_df = production_df.drop(columns=[col for col in existing_prediction_cols if col in production_df.columns], errors='ignore')

    # Feature engineering for live processing
    production_df['Collection Date'] = pd.to_datetime(production_df['Collection Date'], dayfirst=True, errors='coerce')
    today = pd.Timestamp(pd.Timestamp.today().date())
    production_df['days_in_limbo'] = (today - production_df['Collection Date']).dt.days.fillna(0)

    # Rebuild feature matrix array matching training input shape
    processed = pd.DataFrame()
    processed['Cheque Amount'] = pd.to_numeric(production_df['Cheque Amount'], errors='coerce').fillna(0)
    processed['month_encoded'] = production_df['Collection Date'].dt.month.fillna(1)
    processed['days_in_limbo'] = production_df['days_in_limbo']
    processed['agent_risk_score'] = production_df['Agent Name'].map(agent_risk).fillna(overall_avg_risk)

    # Convert dataframe to float32 matrix for ONNX input compatibility
    onnx_input = processed.values.astype(np.float32)

    # 3. ONNX Runtime Inference Execution
    # skl2onnx outputs format: [label_predictions, probabilities_list_of_dicts]
    _, probabilities = session.run(None, {input_name: onnx_input})
    
    # Safely extract Class 1 probabilities (True/Dishonor risk)
    live_probabilities = np.array([p[1] for p in probabilities])
    live_predictions = (live_probabilities >= OPERATIONAL_THRESHOLD).astype(int)

    # 4. Preparing payload transformations to sync back to Sheet
    production_df['Dishonor Probability'] = np.round(live_probabilities * 100, 1).astype(str) + "%"
    production_df['ML prediction'] = np.where(
        live_predictions == 1, "🚨 FLAG: High Risk (Review Transaction)", "✅ PASS: Low Risk"
    )

    # Core Spreadsheet Sync Engine Update
    current_headers = production_sheet.row_values(1)
    for col_name in ['Dishonor Probability', 'ML prediction']:
        if col_name not in current_headers:
            production_sheet.update_cell(1, len(current_headers) + 1, col_name)
            current_headers = production_sheet.row_values(1)

    dishonor_prob_col_idx = current_headers.index('Dishonor Probability') + 1
    ml_prediction_col_idx = current_headers.index('ML prediction') + 1

    cell_updates = []
    for row_idx, val in enumerate(production_df['Dishonor Probability'].tolist(), start=2):
        cell_updates.append(gspread.Cell(row=row_idx, col=dishonor_prob_col_idx, value=val))
    for row_idx, val in enumerate(production_df['ML prediction'].tolist(), start=2):
        cell_updates.append(gspread.Cell(row=row_idx, col=ml_prediction_col_idx, value=val))

    production_sheet.update_cells(cell_updates, value_input_option='USER_ENTERED')
    print(f"🎉 API background processing completed. Synchronized {len(production_df)} entries.")

@app.get("/")
def check_status():
    has_model = os.path.exists(ONNX_MODEL_PATH)
    has_metadata = os.path.exists(METADATA_PATH)
    return {
        "service": "ONNX Inference Engine", 
        "status": "online",
        "artifacts_detected": has_model and has_metadata
    }

@app.post("/predict")
def run_predictions(background_tasks: BackgroundTasks):
    # Immediate safety check to ensure files are present before starting a thread
    if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=503, detail="Model artifact files missing on server container.")
        
    background_tasks.add_task(run_inference_pipeline)
    return {"status": "accepted", "message": "Inference compilation pipeline triggered asynchronously via background worker threads."}