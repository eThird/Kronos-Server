from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Deep Learning Roberta Based Sentiment Analysis
def roberta_predict(sentence):
    encoded_text = tokenizer(sentence, return_tensors="pt")
    output = model(**encoded_text)
    
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    if scores[0] > scores [1] and scores[0] > scores[2]: # Negative
        return "NEG"
    elif scores[1] > scores [2] and scores[1] > scores[0]: # Neutral
        return "NEU"
    elif scores[2] > scores [1] and scores[2] > scores[0]: # Positive
        return "POS"

    else: # Failed
        return "ERR"
