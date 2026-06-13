import os
import json
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# CONFIGURATION
RANDOM_STATE = 42
TEST_SIZE = 0.20
HISTORICAL_SPREADSHEET_ID = "1NIVid942fOylEXsqztqVR1akz4nnEZTmv9ryRfsUKj8"
HISTORICAL_TAB_NAME = "RAW Data"

def get_gspread_client():
    vault_json_string = os.getenv('GCP_SERVICE_ACCOUNT_JSON')
    if not vault_json_string:
        raise ValueError("Missing GCP_SERVICE_ACCOUNT_JSON in environment variables.")
    credentials_dict = json.loads(vault_json_string)
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    scoped_credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
    return gspread.authorize(scoped_credentials)

def run_training_pipeline():
    print("🔄 Fetching historical training data...")
    gc = get_gspread_client()
    workbook_history = gc.open_by_key(HISTORICAL_SPREADSHEET_ID)
    historical_sheet = workbook_history.worksheet(HISTORICAL_TAB_NAME)
    historical_df = pd.DataFrame(historical_sheet.get_all_records())

    # Clean and Target Setup
    historical_df.columns = historical_df.columns.str.strip()
    historical_df['Collection Date'] = pd.to_datetime(historical_df['Collection Date'], dayfirst=True)
    historical_df['Status_Clean'] = historical_df['Status'].fillna('').astype(str).str.strip().str.lower()
    historical_df['target'] = historical_df['Status_Clean'].str.contains('dishonor').astype(int)

    X_train_raw, _, y_train, _ = train_test_split(
        historical_df, historical_df['target'], test_size=TEST_SIZE, stratify=historical_df['target'], random_state=RANDOM_STATE
    )

    # Calculate Risk Profiles
    overall_avg_risk = float(y_train.mean())
    agent_risk = X_train_raw.groupby('Agent Name')['target'].mean().to_dict()

    # Save Metadata needed for exact Feature Engineering during execution
    metadata = {
        "overall_avg_risk": overall_avg_risk,
        "agent_risk": agent_risk
    }
    
    # PATH FIX: Step up one level out of 'train' and into 'prediction service/artifacts'
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prediction service", "artifacts"))
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"✅ Model metadata saved to {metadata_path}.")

    # Feature Engineering
    def engineer_features(df):
        processed = pd.DataFrame(index=df.index)
        processed['Cheque Amount'] = pd.to_numeric(df['Cheque Amount'], errors='coerce').fillna(0)
        processed['month_encoded'] = pd.to_datetime(df['Collection Date'], dayfirst=True, errors='coerce').dt.month.fillna(1)

        if 'Delay' in df.columns:
            processed['days_in_limbo'] = pd.to_numeric(df['Delay'], errors='coerce').fillna(0)
        else:
            processed['days_in_limbo'] = 0

        processed['agent_risk_score'] = df['Agent Name'].map(agent_risk).fillna(overall_avg_risk)
        return processed

    X_train = engineer_features(X_train_raw)

    print("🏋️‍♂️ Training Scikit-Learn Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
    model.fit(X_train.values, y_train.values)

    # Convert Scikit-Learn Model to ONNX Form
    print("🔄 Converting model matrix architecture to ONNX format...")
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

    onnx_path = os.path.join(output_dir, "rf_dishonor_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"🚀 Model saved to {onnx_path}.")

if __name__ == "__main__":
    run_training_pipeline()