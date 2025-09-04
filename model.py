import pandas as pd
#from text import remove_puncts
import nltk

nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))


## To avoid "huggingface_hub.errors.HFValidationError"
## Source : https://stackoverflow.com/questions/76500504/
# from os.path import dirname
# tokenizer = GPT2Tokenizer.from_pretrained(f'{dirname(__file__)}/assets/')
#########################################################

# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset
# import torch

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
    flat_list.append((item["passage"]))
    flat_list.extend([(t[2:]) for t in item["questions"]])

# print(flat_list)
print(len(flat_list))

print([text[0:10] for text in flat_list])
label= [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2] #,3,3,3,3,3,4,4,4,4,4]
df = pd.DataFrame({"text":flat_list,'target': label})
# print(train_dataset["labels"])
print(df.head())

def preprocess_text(text):

    # Lowercase the text

    text = text.lower()

    # Remove punctuation

    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text and remove stopwords

    text = [word for word in text.split() if word not in stop_words]

    return ' '.join(text)


# Apply preprocessing to the text column

df['cleaned_text'] = df['text'].apply(preprocess_text)

print(df[['text', 'cleaned_text']].head())


# Initialize the TF-IDF vectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000)


# Transform the cleaned text to TF-IDF feature matrix

X = tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()


# Define the target variable

y = df['target']

print(X.shape, y.shape)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Logistic Regression model

model = LogisticRegression(max_iter=1000)


# Fit the model on the training data

model.fit(X_train, y_train)

# Make predictions on the test set

y_pred = model.predict(X_test)


# Generate the classification report

print(classification_report(y_test, y_pred))


# Create the confusion matrix

# conf_matrix = confusion_matrix(y_test, y_pred)

# print(conf_matrix)

# Visualize the confusion matrix

# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

# plt.ylabel('Actual')

# plt.xlabel('Predicted')

# plt.title('Confusion Matrix')

# plt.show()
# --- Function to predict passage index from a new question ---
def predict_passage(question):
    # Preprocess (same steps as before)
    cleaned = preprocess_text(question)
    # Convert to TF-IDF
    vector = tfidf_vectorizer.transform([cleaned]).toarray()
    # Predict
    pred_class = model.predict(vector)[0]

    # Map class to passage title
    passage_titles = {
        0: "Passage 1: The lost dog",
        1: "Passage 2: The perfect day for a picnic",
        2: "Passage 3: The science fair project"
    }

    return passage_titles[pred_class]

# --- Example usage ---
new_question = "Where was Buster found?"
print("Predicted passage:", predict_passage('Under which tree did the family sit for their picnic?'))


