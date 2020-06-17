import operator
import numpy as np

from BrunoMars import bruno_docs
from Weekend import weekend_docs
from ElvisPresley import elvis_docs
# import sklearn modules here:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Setting up the combined list of singers' writing samples
singer_docs = bruno_docs + weekend_docs + elvis_docs
# Setting up labels for three singers
# these are number of lines in the data set of the singers
singer_labels = [1] * 154 + [2] * 77 + [3] * 69

# song which will be tested
print("\n\nTest song -->The Weeknd - The Hills\n\n")

mystery_song = """
Remember that time I showed up with just panties under my coat? (Under my coat)
High heels, they was knee high, and my legs was grippin' that throat (grippin' that)
You told me this and I quote (this and I)
'Cause we popped pills and you smoked
You said, your stuff got me strung out
It's like doin' lines of some coke
You always say it's the best that you ever had in your life
And you always play with it good when we be speedin' off in that Wraith
Just keep coppin' them things that you be coppin' me on the eighth
You the president and I'm Biden
Just slide in 'cause you safe, nigga, third base
I only call you when it's half past five
The only time that I'll be by your side
I only love it when you touch me, not feel me
When I'm fucked up, that's the real me
When I'm fucked up, that's the real me, yeah
I only call you when it's half past five
The only time I'd ever call you mine
I only love it when you touch me, not feel me
When I'm fucked up, that's the real me
When I'm fucked up, that's the real me, babe
I'ma let you know and keep it simple (know)
Tryna keep it up don't seem so simple
I just fucked two bitches 'fore I saw you
And you gon' have to do it at my tempo
Always tryna send me off to rehab
Drugs started feelin' like it's decaf
I'm just tryna live life for the moment
And all these motherfuckers want a relapse
I only call you when it's half past five
The only time that I'll be by your side
I only love it when you touch me, not feel me
When I'm fucked up, that's the real me
When I'm fucked up, that's the real me, yeah
I only call you when it's half past five
The only time I'd ever call you mine
I only love it when you touch me, not feel me
When I'm fucked up, that's the real me
When I'm fucked up, that's the real me, babe
Hills have eyes, the hills have eyes
Who are you to judge?
Who are you to judge?
Hide your lies, girl, hide your lies (oh, baby)
Only you to trust, only you (only you, only you)
I only call you when it's half past five
The only time that I'll be by your side
I only love it when you touch me, not feel me
When I'm fucked up, that's the real me
When I'm fucked up, that's the real me, yeah
I only call you when it's half past five
The only time I'd ever call you mine
I only love it when you touch me, not feel me
When I'm fucked up, that's the real me
When I'm fucked up, that's the real me, babe
"""


# Create object of CountVectorizer()
bag_of_words_vectorizer = CountVectorizer()

# Define singer_vectors:
# .fit_transform is used for fit and transform --> Fit to data, then transform it.
# fit --> Compute the mean and std to be used for later scaling.
# transform --> Perform standardization by centering and scaling
singer_vectors = bag_of_words_vectorizer.fit_transform(singer_docs)

# Define mystery_song_vector:
# transform --> Perform standardization by centering and scaling
mystery_song_vector = bag_of_words_vectorizer.transform([mystery_song])

# Define song_classifier:

song_classifier = MultinomialNB()

# Train the classifier:
# fit --> Compute the mean and std to be used for later scaling.
song_classifier.fit(singer_vectors,singer_labels)

# Change predictions:
# predict --> predicts song with highest match
predictions = song_classifier.predict(mystery_song_vector)
probabilities = song_classifier.predict_proba(mystery_song_vector)

if predictions[0] == 1:
    artist = "Bruno Mars"
elif predictions[0] == 2:
    artist = "The Weeknd"
elif predictions[0] == 3:
    artist = "Elvis Presley"

mystery_song = predictions[0] if predictions[0] else "someone else"

print("Analyzer results : ")
print("\n\nThe song was by {}!".format(artist))
print("\n\nProbabilities -> " , probabilities)
