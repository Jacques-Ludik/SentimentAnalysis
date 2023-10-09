# Sentiment Analysis with Movie Reviews

This project focuses on sentiment analysis using movie reviews from the Kaggle dataset. It employs the Bag of Words approach, supplemented with the Random Forest classifier, and also explores sentiment analysis with TextBlob for comparison.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Project Structure

The project consists of three main files:

1. `main.py`: Contains the primary processing and modeling code.
2. `kaggleword2vecUtility.py`: A helper utility to process raw HTML text into segments for further learning. It is sourced from an external GitHub repository.
3. `Evaluate model.py`: Evaluates the accuracy of the model's predictions.

## Setup

1. **Prerequisites**: Ensure you have the following Python libraries installed:

   - os
   - sklearn
   - pandas
   - nltk
   - textblob
   - BeautifulSoup (from bs4)

2. **Dataset**: Download the dataset (not provided here) and place it in a `data` directory in the root folder. The expected files are:
   - `labeledTrainData.tsv`
   - `testData.tsv`
   - `unlabeledTrainData.tsv`

## Usage

1. **Training & Prediction**: Run the `main.py` script to process the dataset, train the Random Forest model, and generate predictions:

   ```bash
   python main.py
   ```

2. **Evaluation**: To evaluate the accuracy of the model, execute the `Evaluate model.py` script:
   ```bash
   python Evaluate\ model.py
   ```

This will display the accuracy of the model based on the Bag of Words approach and the predictions saved in `Bag_of_Words_model.csv`.

## Acknowledgments

- The `KaggleWord2VecUtilityClass` used in this project can be found [here](https://github.com/wendykan/DeepLearningMovies/blob/master/KaggleWord2VecUtility.py).
