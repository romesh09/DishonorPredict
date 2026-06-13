import os
import json
import numpy as np
import pandas as pd
import gspread
import onnxruntime as rt
from google.oauth2.service_account import Credentials

# SPREADSHEET CONFIGURATION
PREDICTION_SPREADSHEET_ID = "18nbA_0B0AHpuAYus-zZbKj_HSYAYpbVTdC83lA_D9ZQ"
PREDICTION_TAB_NAME = "dishonorPredict"
OPERATIONAL_THRESHOLD = 0.10

# Paths relative to the root of your GitHub repository
ARTIFACTS_DIR = os.path.abspath(os.path.join("prediction service", "artifacts"))
ONNX_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rf_dishonor_model.onnx")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

def get_gspread_client():
    # Looks up the secret you just saved in your GitHub settings!
    vault_json_string = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
    if not vault_json_string:
        raise ValueError("Missing GCP_SERVICE_ACCOUNT_JSON environment variable.")
    credentials_dict = json.loads(vault_json_string)
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    return gspread.authorize(Credentials.from_service_account_info(credentials_dict, scopes=scopes))

def run_inference_pipeline():
    if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(METADATA_PATH):
        print(f"❌ Error: Artifact files missing at {ARTIFACTS_DIR}")
        return

    print("🧠 Loading ONNX session and metadata...")
    session = rt.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    overall_avg_risk = metadata["overall_avg_risk"]
    agent_risk = metadata["agent_risk"]

    print("📥 Fetching production rows from Google Sheets...")
    gc = get_gspread_client()
    workbook_prediction = gc.open_by_key(PREDICTION_SPREADSHEET_ID)
    production_sheet = workbook_prediction.worksheet(PREDICTION_TAB_NAME)
    production_df = pd.DataFrame(production_sheet.get_all_records())

    if len(production_df) == 0:
        print("⚠️ Sheet is empty. No active rows to process today.")
        return

    production_df.columns = production_df.columns.str.strip()
    existing_prediction_cols = ['Dishonor Probability', 'ML prediction']
    production_df = production_df.drop(columns=[col for col in existing_prediction_cols if col in production_df.columns], errors='ignore')

    # Feature engineering
    production_df['Collection Date'] = pd.to_datetime(production_df['Collection Date'], dayfirst=True, errors='coerce')
    today = pd.Timestamp(pd.Timestamp.today().date())
    production_df['days_in_limbo'] = (today - production_df['Collection Date']).dt.days.fillna(0)

    processed = pd.DataFrame()
    processed['Cheque Amount'] = pd.to_numeric(production_df['Cheque Amount'], errors='coerce').fillna(0)
    processed['month_encoded'] = production_df['Collection Date'].dt.month.fillna(1)
    processed['days_in_limbo'] = production_df['days_in_limbo']
    processed['agent_risk_score'] = production_df['Agent Name'].map(agent_risk).fillna(overall_avg_risk)

    onnx_input = processed.values.astype(np.float32)

    print("🎯 Running ONNX inference pipeline...")
    _, probabilities = session.run(None, {input_name: onnx_input})
    
    live_probabilities = np.array([p[1] for p in probabilities])
    live_predictions = (live_probabilities >= OPERATIONAL_THRESHOLD).astype(int)

    production_df['Dishonor Probability'] = np.round(live_probabilities * 100, 1).astype(str) + "%"
    production_df['ML prediction'] = np.where(
        live_predictions == 1, "🚨 FLAG: High Risk (Review Transaction)", "✅ PASS: Low Risk"
    )

    print("📤 Syncing predictions back to Google Sheets...")
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
    print(f"🎉 Success! Processed and updated {len(production_df)} entries completely on autopilot.")

if __name__ == "__main__":
    run_inference_pipeline()