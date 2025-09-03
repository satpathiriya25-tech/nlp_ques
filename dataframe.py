import pandas as pd
from test import remove_puncts

## To avoid "huggingface_hub.errors.HFValidationError"
## Source : https://stackoverflow.com/questions/76500504/
from os.path import dirname
tokenizer = GPT2Tokenizer.from_pretrained(f'{dirname(__file__)}/assets/')
#########################################################

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

data=[
    {"title" : "Passage 1: The lost dog",
     "passage": "My neighbor, Mrs. Gable, was frantic. Her beloved beagle, Buster, had gone missing from their yard. He was a mischievous dog, known for his ability to squeeze through small gaps in the fence. A group of us from the neighborhood, including my own family, joined Mrs. Gable to search for him. We split up and walked for hours, calling his name. Finally, my brother spotted him near the creek, his tail wagging happily. Buster was muddy and tired, but otherwise unharmed. Mrs. Gable was so relieved she hugged my brother and gave Buster a big, excited squeeze.",
     "questions": ["1.Why was Mrs. Gable frantic?",
                   "2.What was Buster known for?",
                   "3.Where was Buster found?",
                   "4.How did Buster look when he was found?",
                  ]
},
    {"title": "Passage 2: The perfect day for a picnic",
     "passage":"The sun shone brightly, promising a perfect day. A family decided it was an ideal time for a picnic in the park. They packed a large basket filled with sandwiches, fresh fruit, and cookies. When they arrived, they found a beautiful, shady spot under a large oak tree. A gentle breeze rustled the leaves above them as they spread out their blanket. The sound of children laughing on the playground drifted towards them, and a friendly squirrel even came to visit, hoping for a stray crumb. It was a relaxing and memorable afternoon.",
     "questions": ["1.What did the family pack for their picnic?",
                "2.Where did the family have their picnic?",
                "3.Besides the family, who else visited their picnic?",
                "4.What made the afternoon so relaxing and memorable?",
                 ]
},
    {"title":"Passage 3: The science fair project",
    "passage":"It was the night before the school science fair, and a student was still putting the finishing touches on the project. The experiment was about how different types of soil affect plant growth. Weeks were spent planting seeds in different pots, each with a unique soil composition, and carefully recording observations. The tri-fold display board was covered in charts and photos of the growing plants. The final step was to write the conclusion, summarizing the findings. As the last sentence was typed, a wave of relief and pride was felt. It might not win a prize, but the student was proud of the hard work.",
     "questions": ["1.What was the science fair project about?",
                "2.How did the student record observations for the experiment?",
                "3.What was the final step for the student to complete the project?",
                "4.Why did the student feel pride about the project?",
                 ]
    }

]

df=pd.DataFrame(data)

# df.to_csv("Question_dataset.csv",index= False)
# print(df)

df.shape
flat_list = []
for item in data:
    # flat_list.append(item["title"])
    flat_list.append(remove_puncts(item["passage"]))
    flat_list.extend([remove_puncts(t[2:]) for t in item["questions"]])

# print(flat_list)
print(len(flat_list))

print([text[0:10] for text in flat_list])
label= [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2] #,3,3,3,3,3,4,4,4,4,4]
train_dataset = Dataset.from_dict({"text":flat_list,"labels":label})
# print(train_dataset["labels"])
print(train_dataset)
model_name = 'xlm-robert-base'
tokenizer = AutoTokenizer.from_pretrained(train_dataset)
tokenized_train_dataset = tokenizer(train_dataset)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)
trainer.train()
test_questions = ["Who spotted Buster near the creek?","What kind of tree did the family sit under during their picnic?","What covered the student’s tri-fold display board at the science fair?"]
encodings = tokenizer(test_questions, return_tensors="pt", padding=True, truncation=True)
outputs = model(**encodings)
preds = torch.argmax(outputs.logits, dim=1)
print("Predicted labels:", preds.tolist()) 






