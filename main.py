import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import nltk
from KaggleWord2VecUtility import KaggleWord2VecUtilityClass
from textblob import TextBlob

# if __name__ == '__main__':
# Read the data
train = pd.read_csv(os.path.join(os.path.dirname(
    __file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__),
                   'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(
    __file__), 'data', "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)
print("The first review is:")
print(train["review"][0])
input("Press Enter to continue...")

# [2] Clean the training and test sets
# print("Download text data sets.")
# nltk.download()  # Download text data sets, including stop words

clean_train_reviews = []
print("Cleaning and parsing the training set movie reviews...\n")

for i in range(0, len(train["review"])):
    clean_train_reviews.append(
        " ".join(KaggleWord2VecUtilityClass.review_to_wordlist(train["review"][i], True)))

print("Creating the bag of words...\n")
vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                             preprocessor=None, stop_words=None, max_features=5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print("Training the random forest (this may take a while)...")
forest = RandomForestClassifier(n_estimators=1000)
forest = forest.fit(train_data_features, train["sentiment"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0, len(test["review"])):
    clean_test_reviews.append(
        " ".join(KaggleWord2VecUtilityClass.review_to_wordlist(test["review"][i], True)))

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

print("Predicting test labels...\n")
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv(os.path.join(os.path.dirname(__file__), 'data',
                           'Bag_of_Words_model.csv'), index=False, quoting=3)
print("Wrote results to Bag_of_Words_model.csv")

# Textblob sentiment analysis to compare
predicted_sentiments = []

for review in clean_test_reviews:
    analysis = TextBlob(review)
    # TextBlob returns polarity in the range [-1, 1].
    # We'll classify reviews with polarity > 0 as positive (sentiment = 1)
    if analysis.sentiment.polarity > 0:
        predicted_sentiments.append(1)
    else:
        predicted_sentiments.append(0)

output = pd.DataFrame(
    data={"id": test["id"], "sentiment": predicted_sentiments})
output.to_csv(os.path.join(os.path.dirname(__file__), 'data',
              'TextBlob_Predictions.csv'), index=False, quoting=3)

print("Wrote results to TextBlob_Predictions.csv")

"""# [3] Evaluate the model
# 1. Load the CSV file into a DataFrame
df = pd.read_csv('Bag_of_Words_model.csv')

# 2. Extract the ratings from the `id` column
df['rating'] = df['id'].str.split('_').str[-1].astype(int)

# 3. Compute the predicted sentiment based on the extracted ratings
df['predicted_sentiment'] = df['rating'].apply(lambda x: 1 if x >= 5 else 0)

# 4. Compare the predicted sentiment with the actual sentiment to compute the accuracy
correct_predictions = (df['sentiment'] == df['predicted_sentiment']).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions * 100

print(f'Accuracy: {accuracy:.2f}%')"""
