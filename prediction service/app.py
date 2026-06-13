import os
import json
import numpy as np
import pandas as pd
import gspread
import onnxruntime as rt
from fastapi import FastAPI, HTTPException, BackgroundTasks
from google.oauth2.service_account import Credentials

app = FastAPI(title="RA16 ONNX Live Inference Service")

PREDICTION_SPREADSHEET_ID = "18nbA_0B0AHpuAYus-zZbKj_HSYAYpbVTdC83lA_D9ZQ"
PREDICTION_TAB_NAME = "dishonorPredict"
OPERATIONAL_THRESHOLD = 0.10

def get_gspread_client():
    vault_json_string = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
    if not vault_json_string:
        raise HTTPException(status_code=500, detail="Missing GCP_SERVICE_ACCOUNT_JSON variable.")
    credentials_dict = json.loads(vault_json_string)
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    return gspread.authorize(Credentials.from_service_account_info(credentials_dict, scopes=scopes))

def run_inference_pipeline():
    if not os.path.exists("rf_dishonor_model.onnx") or not os.path.exists("model_metadata.json"):
        raise HTTPException(status_code=503, detail="Model artifact files not found on server.")

    # 1. Load ONNX Session & Training Metadata mapping
    session = rt.InferenceSession("rf_dishonor_model.onnx")
    input_name = session.get_inputs()[0].name
    
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    overall_avg_risk = metadata["overall_avg_risk"]
    agent_risk = metadata["agent_risk"]

    # 2. Get Live Production Rows
    gc = get_gspread_client()
    workbook_prediction = gc.open_by_key(PREDICTION_SPREADSHEET_ID)
    production_sheet = workbook_prediction.worksheet(PREDICTION_TAB_NAME)
    production_df = pd.DataFrame(production_sheet.get_all_records())

    if len(production_df) == 0:
        return "No rows to process inside the sheet today."

    # Clean headers
    production_df.columns = production_df.columns.str.strip()
    existing_prediction_cols = ['Dishonor Probability', 'ML prediction']
    production_df = production_df.drop(columns=[col for col in existing_prediction_cols if col in production_df.columns], errors='ignore')

    # Feature engineering for live data
    production_df['Collection Date'] = pd.to_datetime(production_df['Collection Date'], dayfirst=True, errors='coerce')
    today = pd.Timestamp(pd.Timestamp.today().date())
    production_df['days_in_limbo'] = (today - production_df['Collection Date']).dt.days.fillna(0)

    # Rebuild feature matrix matching model expectations
    processed = pd.DataFrame()
    processed['Cheque Amount'] = pd.to_numeric(production_df['Cheque Amount'], errors='coerce').fillna(0)
    processed['month_encoded'] = production_df['Collection Date'].dt.month.fillna(1)
    processed['days_in_limbo'] = production_df['days_in_limbo']
    processed['agent_risk_score'] = production_df['Agent Name'].map(agent_risk).fillna(overall_avg_risk)

    # Convert dataframe values to float32 matrix array for ONNX input evaluation
    onnx_input = processed.values.astype(np.float32)

    # 3. ONNX Inference Execution
    # onnx outputs structures: [label_predictions, probabilities_list_dict]
    _, probabilities = session.run(None, {input_name: onnx_input})
    
    # Extract the probabilities specifically for Class 1 (Dishonor flag)
    live_probabilities = np.array([p[1] for p in probabilities])
    live_predictions = (live_probabilities >= OPERATIONAL_THRESHOLD).astype(int)

    # 4. Preparing data to write back to sheet
    production_df['Dishonor Probability'] = np.round(live_probabilities * 100, 1).astype(str) + "%"
    production_df['ML prediction'] = np.where(
        live_predictions == 1, "🚨 FLAG: High Risk (Review Transaction)", "✅ PASS: Low Risk"
    )

    # Core Spreadsheet Sync Engine
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

    production_sheet.update_cells(cell_updates)
    return f"Live synchronization successfully processed {len(production_df)} active rows."

@app.get("/")
def check_status():
    has_model = os.path.exists("rf_dishonor_model.onnx")
    return {"service": "ONNX Inference Engine", "model_loaded_and_ready": has_model}

@app.post("/predict")
def run_predictions(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_inference_pipeline)
    return {"status": "accepted", "message": "Inference run triggered using background tasks."}