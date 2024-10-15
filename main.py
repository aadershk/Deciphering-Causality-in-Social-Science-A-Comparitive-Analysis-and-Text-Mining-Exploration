import os
from pipeline_design import CausalExtractionPipeline

input_folders = [
    "/content/drive/MyDrive/Coda_PDF",
    "/content/drive/MyDrive/Xavier_PDF",
    "/content/drive/MyDrive/ThirdDataset_PDF"
]
output_folder = "/content/drive/MyDrive/processed_csv_causal_new"
os.makedirs(output_folder, exist_ok=True)

causal_pipeline = CausalExtractionPipeline()

for folder in input_folders:
    dataset_id = os.path.basename(folder)
    for pdf_file in os.listdir(folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder, pdf_file)

            output_csv = os.path.join(output_folder, f"causal_{os.path.splitext(pdf_file)[0]}.csv")

            causal_pipeline.process_dataset_pipeline(dataset_id, pdf_path, output_csv)

print("Execution completed. Check the 'processed_csv_causal' directory for results.")
