# Shazam-for-lyrics

In this project I used the concept of BAG-OF-WORDS LANGUAGE MODEL to find the artist of the song given a song's lyrics.
There is a database for three artists i.e. Bruno Mars , The Weeknd and Elvis Presley. The data is trained and transformed using use scikit-learn’s bag-of-words and Naive Bayes classifier. 

BAG-OF-WORDS LANGUAGE MODEL : In this model, a text is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity. 

Libraries used for the project:
* sklearn -> Simple and efficient tools for predictive data analysis
  > MultinomialNB : The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word       counts for text classification). 
