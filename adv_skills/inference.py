import torch
from lightning import Trainer
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from pretrained import BertNLIFinetuner

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertNLIFinetuner.load_from_checkpoint('bert_nli.ckpt')
trainer = Trainer(accelerator="auto", devices=1)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class NLIPredictDataset(Dataset):
    def __init__(self, pairs):
        # pairs is a list of (premise, hypothesis)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        premise, hypothesis = self.pairs[idx]

        encoded = tokenizer(
            premise,
            hypothesis,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded["token_type_ids"].squeeze(0),
        }


pairs = [
    ("They are playing outside", "They are having lunch at home"),
    ("The cat is on the mat", "The animal is on the mat"),
    ("He is riding a bike", "He is sleeping"),
    ("It is sunny", "I walk in the park"),
]

dataset = NLIPredictDataset(pairs)
predict_loader = DataLoader(dataset, batch_size=1)
predictions = trainer.predict(model, dataloaders=predict_loader)

preds = [p.item() for p in predictions]
class_names = ["contradiction", "neutral", "entailment"]

for (premise, hypothesis), pred_id in zip(pairs, preds):
    label = class_names[pred_id]
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Prediction: {label}")
    print("-" * 40)
