#data_prep.py
#!!!THIS IS NOT MY CODE, KEEP THA IN MIND WHEN GIVING CREDIT!!!
#source: https://aws.amazon.com/blogs/machine-learning/fine-tune-and-deploy-the-protbert-model-for-protein-classification-using-amazon-sagemaker/

import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset

class ProteinSequenceDataset(Dataset):
    def __init__(self, sequence, targets, tokenizer, max_len):
        self.sequence = sequence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'protein_sequence': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
# import torch
import torch.nn.functional as F
# import torch.nn as nn

PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(self.bert.config.hidden_size, n_classes),
                                        nn.Tanh())
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        return self.classifier(output.pooler_output)

# SageMaker Distributed code.
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
import smdistributed.dataparallel.torch.distributed as dist

# intializes the process group for distributed training
dist.init_process_group()

from sagemaker.pytorch import PyTorch

TRAINING_JOB_NAME="protbert-training-pytorch-{}".format(time.strftime("%m-%d-%Y-%H-%M-%S")) 
print('Training job name: ', TRAINING_JOB_NAME)

estimator = PyTorch(
    entry_point="train.py",
    source_dir="code",
    role=role,
    framework_version="1.6.0",
    py_version="py36",
    instance_count=1,  # this script support distributed training for only GPU instances.
    instance_type="ml.p3.16xlarge",
    distribution={'smdistributed':{
        'dataparallel':{
            'enabled': True
        }
       }
    },
    debugger_hook_config=False,
    hyperparameters={
        "epochs": 3,
        "num_labels": num_classes,
        "batch-size": 4,
        "test-batch-size": 4,
        "log-interval": 100,
        "frozen_layers": 15,
    },
    metric_definitions=[
                   {'Name': 'train:loss', 'Regex': 'Training Loss: ([0-9\\.]+)'},
                   {'Name': 'test:accuracy', 'Regex': 'Validation Accuracy: ([0-9\\.]+)'},
                   {'Name': 'test:loss', 'Regex': 'Validation loss: ([0-9\\.]+)'},
                ]
)
estimator.fit({"training": inputs_train, "testing": inputs_test}, job_name=TRAINING_JOB_NAME)