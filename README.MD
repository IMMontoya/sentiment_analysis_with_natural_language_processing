# Sentiment Analysis of IMDB Movie Reviews

[Link To Notebook](/nlp_analysis_movie_reviews.ipynb)

## Project Overview

The goal of this project is to develop a machine learning model that can automatically classify IMDB movie reviews as either positive or negative. This involves using a dataset of IMDB reviews labeled by sentiment to train and evaluate various classification models.

### Project Goals:

- Achieve an F1 score of at least 0.85 in classifying movie reviews.
- Explore and preprocess the data to address class imbalances and prepare it for modeling.
- Train and compare the performance of at least three different models.
- Conduct an analysis of the models' performance and present the findings.

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Model Training and Evaluation](#model-training-and-evaluation)
4. [Conclusion](#conclusion)
5. [Future Work](#future-work)

## Data Preprocessing

A few steps were taken to optimize the dataset for this project's purposes:

- The original dataset contained information on other media types such as TV series, video games, and more. Only records of title type "movie" and "tvMovie" were kept in order to stay within the project's scope.

- Duplicate reviews were dropped from the dataset.

- **Note:** the goal of this project was to predict sentiment based exclusively on the text within the review, thus other features of the dataset were not part of the training features.

- Reviews were normalized and lemmatized with the function `normalize_text`

    ```python
    def normalize_text(text):
        """
        Normalize and lemmatize the given text by converting it to lowercase, removing punctuation, digits, and special characters,
        and getting rid of extra spaces.
        
        Args:
            text (str): The text to be normalized.
        
        Returns:
            str: The normalized text.
        """
        # Lowercase
        norm_text = text.lower()
        # Remove punctuation digits and special characters
        norm_text = re.sub(r'[^a-z\']', " ", norm_text)
        # Get rid of extra spaces
        norm_text = norm_text.split()
        norm_text = " ".join(norm_text)
        
        # Lemmatize
        doc = nlp(norm_text)
        
        lemas = []
        for token in doc:
            lemas.append(token.lemma_)
            
        norm_text = " ".join(lemas)
        
        # nlp capitalizes pronouns, so we need to lowercase them
        norm_text = norm_text.lower()
        
        return norm_text
    ```

## Exploratory Data Analysis

### Visualizing Movies and Reviews per Year

![Movies and Reviews per Year](/images/movies_reviews_per_year.png)

As one would expect, there are more movies in recent years. Oddly, there is a dip movies between 1960 and 1967. The sharp decline around 2006 suggest that the data was probably collected sometime in 2010. On the second plot we can see that the distribution of positive to negative reviews is about 50/50 throughout, while the number of reviews per movie rises sharply for movies released around 1994.

### Visualizing Reviews Distribution

![Reviews Per Movie Distribution](/images/reviews_per_movie_dist.png)

Most movies have very few reviews. However, some movies have 30 or more reviews. This makes sense: "blockbuster" movies will get a lot of reviews, while most other movies will only be reviewed by a few people. Essentially, you can expect the number of reviews per movie to decline until the film reaches a certain tipping point of popularity where many people submit a review.

### Balance of Train Set VS Test Set

![Ratings balance](/images/trainVtest_ratings.png)
![Sentiment balance](/images/trainVtest_years_sentiment.png)
We can see the train and test datasets are well balanced in terms of their share of positive and negative reviews with respect to `start_year` and the number of reviews per movie. Thankfully the team provided well balanced data, and no additional steps need to be taken to balance the sets or target variable.

## Model Training and Evaluation

Four (4) models were considered for this task:

### Baseline (dummy) classifier model that predicts using the `stratified` strategy

![Baseline Model Evaluation](/images/model_performances/dummy.png)
Performance is consistent with what we would expect for random guessing: 50% chance on each prediction.

### Linear regression model(random_state=12345)

![Linear Regression Evaluation](/images/model_performances/lr.png)
Performs quite well. It returns an F1 score of .88 on the test set (the successful model threshold is an F1 score of .85). We can also see that the ideal thresholds for the train and tests sets are very near eac other, indicating consistent performance.  

The ROC curve for both the training and test sets is very close to the top left corner, indicating high true positive rates and low false positive rates. This suggests that the model is excellent at distinguishing between positive and negative reviews, with very few incorrect classifications.  

The Precision-Recall curves for both sets remain near the top-right corner, indicating high precision and recall across various thresholds. This means the model consistently makes accurate positive predictions while also capturing most of the actual positive cases, demonstrating robust performance in handling class imbalance.

### Random forrest classifier model(n_estimators=25, random_state=12345)

![Random Forrest Evaluation](/images/model_performances/rf.png)
The Random Forest model demonstrates excellent performance on the training data, achieving perfect scores across F1, ROC AUC, and APS metrics, which indicates it is overfitting to the training set. In contrast, its performance on the test data is notably lower, with an F1 score of 0.82, a ROC AUC of 0.89, and an APS of 0.87.  

This disparity between the training and test set performances suggests the model fails to generalize well to unseen data. Furthermore, compared to the Linear Regression model, the Random Forest model's lower ROC AUC and APS scores on the test set confirm that it is less effective at distinguishing between positive and negative reviews and maintaining a high precision and recall balance.

### LightGMB classifier model(random_state=12345)

![LightGBM Evaluation](/images/model_performances/lgbm.png)
The LightGBM model performs well enough to meet the threshold of the task, with similar but improved performance compared to the random forrest model. The improved scores on the test set, along with a smaller gap between the training and test sets, reveals this model generalizes to unseen data better than the random forrest.

## My Reviews

Each model was further tested with a small dataset of 8 additional reviews. The results of each models performance on this dataset is summarized in the table:

| Model                | Accuracy (Data) | Accuracy (My Reviews) | F1 (Data) | F1 (My Reviews) | APS (Data) | APS (My Reviews) | ROC AUC (Data) | ROC AUC (My Reviews) |
|----------------------|-----------------|-----------------------|-----------|-----------------|------------|------------------|----------------|----------------------|
| Dummy                | 0.50            | 0.38                  | 0.50      | 0.29            | 0.50       | 0.46             | 0.50           | 0.38                 |
| Logistic Regression  | 0.88            | 0.75                  | 0.88      | 0.67            | 0.95       | 1.00             | 0.95           | 1.00                 |
| Random Forest        | 0.81            | 0.75                  | 0.81      | 0.80            | 0.87       | 0.73             | 0.89           | 0.72                 |
| LightGBM             | 0.86            | 0.75                  | 0.86      | 0.80            | 0.93       | 1.00             | 0.94           | 1.00                 |

## Conclusion

**On the provided data set:**  
The Logistic Regression model and LightGBM model both performed well enough to meet the F1 score threshold of .85 set by the client. The Random forrest model did not clear the threshold. No hyper parameter tunning was performed with any model, and I suspect that the LightGBM model would benefit more from tuning while the Logistic Regression model probably wouldn't. This can be explored more in later iterations.

**On my reviews:**  
While the Logistic Regression and LightGBM models performed significantly worse on the `my_reviews` dataset, it is important to consider that this is an exceedingly small dataset of only 8 records. Additionally, the average length of these reviews were far less than the average length of the train or test sets - indicating that model probably performs better with longer reviews. Intuitively, this makes sense, longer reviews means more tokens to vectorize and draw connections from. This hypothesis could also be explored in a followup to this project. 

## Future Work

The Film Junky Union should consider further exploring Logistic Regression and LightGBM with hyperparameter tuning to see if a more accurate model can be achieved.
