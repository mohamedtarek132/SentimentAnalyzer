import string
import emot
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

tfidf_vectorizer_lemmatizer = joblib.load("tfidf.pkl")

def remove_hashtags(text):
  text = text.split()
  text = [word for word in text if "#" not in word]
  text = ' '.join(text)
  return text

def remove_stop_words(text):
  stop_words = set(stopwords.words('english'))
  text = text.split()
  text = [word for word in text if word not in stop_words]
  text = ' '.join(text)
  return text

def remove_punctuation(text):
  punctuation = string.punctuation
  text = text.translate(str.maketrans("","",punctuation))
  return text

def replace_emojies(text):
  emojies_to_words = {value : key.replace(":","") for key,value in emot.EMOJI_UNICODE.items()}
  emojies_to_words['❤️'] = "heavy_black_heart"
  text = text.split()
  text = [emojies_to_words[word] if word in emojies_to_words.keys() else word for word in text ]
  text = ' '.join(text)
  return text

def change_to_orgin_lemmatizer(text):
  lemmatizer = WordNetLemmatizer()
  wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
  text = text.split()
  text = nltk.pos_tag(text)
  text = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in text]
  text = ' '.join(text)
  return text

def remove_dash(text):
  text = text.translate(str.maketrans("","","'"))
  return text

def preprocessing2(text):
  text = text.strip()
  text = text.lower()
  text = remove_hashtags(text)
  text = remove_punctuation(text)
  text = remove_stop_words(text)
  text = remove_dash(text)
  text = replace_emojies(text)
  text = change_to_orgin_lemmatizer(text)
  text = tfidf_vectorizer_lemmatizer.transform(list([text, "hell"]))
  text = text.toarray()
  return text[0].reshape(1, -1)