from pdf_extractor import extract_pdf_text
from preprocessing_pipeline import preprocess_sentences_parallel
from sentence_splitter import split_into_sentences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv

class CausalExtractionPipeline:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("rasoultilburg/ssc_bert")
        self.model = AutoModelForSequenceClassification.from_pretrained("rasoultilburg/ssc_bert")
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate_causality(self, sentences):
        causal_sentences = []
        batch_size = 128 

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(**tokens)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

            for sentence, pred in zip(batch, predictions):
                if pred == 1:
                    causal_sentences.append(sentence)

        return causal_sentences

    def process_dataset_pipeline(self, dataset_id, pdf_path, output_csv):
        document_id = os.path.basename(pdf_path)

        text = extract_pdf_text(pdf_path)

        sentences = split_into_sentences(text)

        preprocessed_sentences = preprocess_sentences_parallel(sentences)

        causal_sentences = self.evaluate_causality(preprocessed_sentences)

        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["dataset_id", "document_id", "causal_sentence"])
            for sentence in causal_sentences:
                writer.writerow([dataset_id, document_id, sentence])
