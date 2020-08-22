#import nltk
#from nltk.corpus import gutenberg

#https://towardsdatascience.com/introduction-to-natural-language-processing-for-text-df845750fb63

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

with open("test.txt", "r") as file:
    documents = file.read().splitlines()
print(documents)

# Step 2. Design the Vocabulary
# The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()

# Step 3. Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform(documents)

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
print(pd.DataFrame(bag_of_words.toarray(), columns=feature_names))

#USING TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(documents)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
print(pd.DataFrame(values.toarray(), columns = feature_names).to_string())
print(type(pd.DataFrame(values.toarray(), columns = feature_names)))