from transformers import RagRetriever, RagTokenizer, RagSequenceForGeneration
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
import torch

RAG_MODEL_NAME = "facebook/rag-sequence-nq"
MODEL_NAME = "bert-base-uncased"

rag_retriever = RagRetriever.from_pretrained(
    RAG_MODEL_NAME,
    index_name="exact",
    trust_remote_code=True
)

rag_tokenizer = RagTokenizer.from_pretrained(
    RAG_MODEL_NAME,
    trust_remote_code=True
)

rag_model = RagSequenceForGeneration.from_pretrained(
    RAG_MODEL_NAME,
    retriever=rag_retriever,
    trust_remote_code=True
)

num_classes = 2
bert_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes
)

dataset_name = "wiki_dpr"
dataset = load_dataset(dataset_name, trust_remote_code=True)

def tokenize_input(input_text):
    return rag_tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)

input_text = "What is the capital of France?"
tokenized_input = tokenize_input(input_text)

outputs = rag_model.generate(
    input_ids=tokenized_input["input_ids"],
    attention_mask=tokenized_input["attention_mask"],
    num_beams=1,
    max_length=50
)

decoded_output = rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Generated Answer:", decoded_output[0])

bert_inputs = rag_tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)
logits = bert_model(**bert_inputs).logits
predicted_class = torch.argmax(logits, dim=1).item()
print("Predicted Class:", predicted_class)
