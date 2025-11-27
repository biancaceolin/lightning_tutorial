import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from lightning import LightningModule


# I try using a pretrained model (BERT) to perform sentence pairs classification.
# I also add the arg parse for giving hyperparameters from command line for the trainer

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training")
# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TinyNLIDataset(Dataset):
    def __init__(self, split='train'):
        #prints an example input array 
        self.example_input_array = (
            torch.randint(0, 20000, (2, 64)),  # input_ids
            torch.ones(2, 64, dtype=torch.long),  # attention_mask
            torch.zeros(2, 64, dtype=torch.long)  # token_type_ids
        )

        data = [
            # premise, hypothesis, label
            ("The cat is on the mat", "The animal is on the mat", 2),  # entailment
            ("The sky is blue", "The sky is green", 0),               # contradiction
            ("I like pizza", "I eat pizza sometimes", 2),             # entailment
            ("He is driving a car", "He is sleeping", 0),             # contradiction
            ("She reads a book", "She is holding something", 2),      # entailment
            ("They are playing football", "They are watching TV", 0), # contradiction
            ("He has a dog", "He has a pet", 2),                      # entailment
            ("She is cooking", "She might be hungry", 1),             # neutral
            ("It is raining", "The ground might get wet", 1),         # neutral
            ("He is running", "He is moving quickly", 2),             # entailment
        ]
        # classic 80-20 train-val split
        if split == 'train':
            self.data = data[:8]
        else:
            self.data = data[8:]
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        premise, hypothesis, label = self.data[idx]
        encoded = tokenizer(premise, hypothesis, padding='max_length',
                         truncation=True, max_length=64, return_tensors='pt') # pt returns pytorch tensors of shape (batch, seqlen) so you have to squeeze
        item ={ "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded["token_type_ids"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
              }
        return item
    
train_dataset = TinyNLIDataset(split='train')
val_dataset = TinyNLIDataset(split='val')

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Define the model with LightningModule
class BertNLIFinetuner(LightningModule):
    def __init__(self, lr: float = 2e-5, num_classes: int = 3):  #colon suggests type, then write default value
        super().__init__()
        self.save_hyperparameters()
  
        # 1. Load pretrained BERT, base cased version
        self.bert = BertModel.from_pretrained(
            "bert-base-cased",
            output_attentions=True
        )
        # 2. Add the Classification head!!
        #hidden size is the size of the embeddings that BERT produces (of the cls summary token)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes) #you give to Linear the input size and output size
        # 3. Define loss function as cross entropy 
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids): #forward process
            # 4. Forward pass through BERT and classification head
            #breakpoint() very useful for debugging
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            # Get the [CLS] token representation
            last_hidden_state = outputs.last_hidden_state     # [batch, seq_len, hidden]

            cls_hidden = last_hidden_state[:, 0]              # CLS token representation
            logits = self.classifier(cls_hidden)              # [batch, num_classes]
            return logits
        
    def training_step(self, batch, batch_idx):      #loss calculation
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]   


            logits = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"]
            )
            loss = self.loss_fn(logits, labels)
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()

            #log metrics in the progress bar
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc", acc, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]

        logits = self (
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"]
        )

        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True) 
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"]
        )
        
        preds = logits.argmax(dim=-1)
        return preds
    
if __name__ == "__main__":
    from lightning import Trainer, seed_everything
    from lightning.pytorch.profilers import AdvancedProfiler

    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    seed_everything(42)

    model = BertNLIFinetuner(lr=2e-5, num_classes=3)


    #TRAINER PARAMS
    #fast_dev_run=True -> useful for testing quickly if the flow works
    #num_sanity_val_steps= -> runs 2 val steps before training to catch bugs
    #profiler = "simple" to see time for each module, "advanced" for every function!! catch bottlenecks
    # limit_train_batches=0.25, limit_val_batches=0.01 use only 10% of training data and 1% of val data 
    # limit_train_batches=10, limit_val_batches=5 - use 10 batches of train and 5 batches of val
    #checkpoint_callback=True/False - enable/disable model checkpointing, later use torch.load to load weights

    trainer = Trainer(profiler=profiler,max_epochs=3, accelerator="gpu", devices=args.devices, log_every_n_steps=1)

    #ckpt_path="path/to/your/checkpoint.ckpt" #to resume from checkpoint
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint("bert_nli.ckpt")




    '''# Example 

    model.eval()
    sample_premise = "They are playing outside"
    sample_hypothesis = "They are having lunch at home"  
    encoded_sample = tokenizer(sample_premise, sample_hypothesis, padding='max_length',
                            truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        logits = model(
            input_ids=encoded_sample["input_ids"],
            attention_mask=encoded_sample["attention_mask"],
            token_type_ids=encoded_sample["token_type_ids"]
        )
    predicted_class = logits.argmax(dim=-1).item()
    class_names = ["contradiction", "neutral", "entailment"]
    print(f"Predicted class: {class_names[predicted_class]}") '''