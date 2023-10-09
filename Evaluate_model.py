import pandas as pd

# 1. Load the CSV file into a DataFrame
df = pd.read_csv('data/Bag_of_Words_model.csv')

# 2. Extract the ratings from the `id` column
df['rating'] = df['id'].str.split('_').str[-1].astype(int)

# 3. Compute the predicted sentiment based on the extracted ratings
df['predicted_sentiment'] = df['rating'].apply(lambda x: 1 if x >= 5 else 0)

# 4. Compare the predicted sentiment with the actual sentiment to compute the accuracy
correct_predictions = (df['sentiment'] == df['predicted_sentiment']).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions * 100

print(f'Accuracy: {accuracy:.2f}%')
