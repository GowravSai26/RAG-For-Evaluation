import pandas as pd
import json
import requests
from typing import List, Dict, Any

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# -------------------------------
#        READ EXCEL
# -------------------------------

file_path = r"..\data\RAG Evals.xlsx"

sheets = pd.read_excel(file_path, sheet_name=None)

print("Available Sheets:")
sheet_names = list(sheets.keys())

for i, name in enumerate(sheet_names, start=1):
    print(f"{i}. {name}")

choice = int(input("\nEnter sheet number to use: "))
sheet_name = sheet_names[choice - 1]
df = sheets[sheet_name]

print("\nAll Columns in Sheet:")
print(df.columns)

# Pick required columns
selected_columns = ["Group_id", "Question", "Json"]
extracted_df = df[selected_columns]

print("\nExtracted First Row:")
print(extracted_df.head(1))

# Process ONLY row 1 as instructed
row = extracted_df.iloc[0]

my_company = row["Group_id"]
my_question = row["Question"]

# Read JSON column
raw_files = json.loads(row["Json"]) if not pd.isna(row["Json"]) else None

if raw_files is None:
    print("This row has no JSON. Cannot call API.")
    exit()


# -------------------------------
#    PREPARE REQUEST BODY
# -------------------------------

def create_search_request_body(
    company_name: str,
    question: str,
    files_json: List[Dict[str, str]],
    top_k: int = 5,
    mmr: bool = False,
    search_type: str = "semantic"
) -> Dict[str, Any]:

    payload = {
        "client_id": company_name,
        "files": files_json,
        "question": question,
        "top_k": top_k,
        "mmr": mmr,
        "search_type": search_type
    }
    return payload


request_body = create_search_request_body(my_company, my_question, raw_files)


# -------------------------------
#    CALL SEARCH/INGEST API
# -------------------------------

def call_search_api(request_body: dict):

    url = "http://127.0.0.1:8000/search_or_ingest"  # correct backend endpoint

    headers = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    try:
        response = requests.post(url, json=request_body, headers=headers)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as err:
        if response.status_code == 422:
            print(f"Validation Error (422): {response.text}")
        else:
            print(f"HTTP Error occurred: {err}")
        return None

    except Exception as err:
        print(f"An error occurred: {err}")
        return None


# Call the API
api_response = call_search_api(request_body)

if api_response is None:
    print("API failed.")
    exit()

# Extract excerpts
excerptList = [item["excerpt"] for item in api_response["search_results"]]

print("\nAPI Output (excerptList):")
print(excerptList)


# -------------------------------
#   WRITE BACK TO EXCEL
# -------------------------------

# Add a new column to original sheet
df.loc[0, "API_Output"] = "\n\n".join(excerptList)

# Save as a new file
output_path = r"..\data\RAG Evals (OUTPUT).xlsx"
df.to_excel(output_path, index=False)

print(f"\nOutput written to column 'API_Output'")
print(f"Saved file: {output_path}")
print(api_response)
