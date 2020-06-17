LANGUAGE DETECTOR

Hi everyone! This is my own independent NLP/Machine Learning project, a language detection model. Its goal is: given a string of text, identify the language of the string. This is a classic machine learning classification problem, and is a staple in natural language processing (NLP).

The first thing I did was come up with my classes; these are the specific languages I test for. I chose 4 Germanic languages and 4 Romance languages, all in the Latin script. My idea here was that if the model can distinguish between closely related languages, the extracted features are effective. The languages are as follows:

DE (German)

EN (English)

NL (Dutch)

SV (Swedish)

ES (Spanish)

FR (French)

IT (Italian)

PT (Portuguese)

Next, I gathered my training data. For this, I went to Wikipedia and copied 3 articles in all 8 languages into plain text files. I needed to find articles which were sufficiently long in all languages. I also wanted to find articles that discussed a wide variety of topics, so that the words and sentences of my training data could be as diverse as possible. These articles are: "Association Football", "World War 2", and "Atlantic Ocean". 24 total Wikipedia articles therefore serve as my training data. I did some preliminary processing on these articles; I eliminated all the numbered links that show up in the articles, eliminated superfluous information such as links to other articles, left out the cited references sections, and converted everything to lower case. It was straightforward to do this pre-processing from the UNIX command line.

Next, I had to organize my text data in a way that can be fed into and interpreted by a machine learning model. In NLP, the common strategy is to convert to one-hot vectors, where for each individual training example (string of text), a vector representation is given, indicating the number of occurrences of each feature in the entire data set. The "features" in basic NLP algorithms are individual words. However, this is language detection, and so I felt that not only words, but also orthographic sequences were of the utmost importance when determining what language a string o text belongs to. Therefore, I went through the entire data set, and extracted character ngram features from all the articles. I extracted character ngrams of 1-5, as well as all individual words, from the training set. I previously attempted to extract word ngrams from the data, but this did not really help, so I opted for just individual words and character ngrams.

I then used NLTK's sentence tokenizing algorithm to split the articles into individual sentences, keeping their ground truth labels attached to them depending on what specific file each training example came from. Each of thes sentences was then converted to a one-hot vector indicating how many times each word or character ngram occurred within that sentence. The full set of one-hot vectors was then fed into a machine learning classifier, and trained. I attempted 2 different classifiers here, both from scikit-learn: Logistic Regression and Multinomial Naive Bayes.

I also gathered a test set, which was from a very different distribution than the training set. I used a random sentence generator I found online, and generated 80 distinct sentences in English. I then put all of these through Google Translate to get the sentences in the other 7 languages. I put all of these into a text file, with ground truth labels attached, to use as my test set. My idea for this was that if the model can predict the language accurately for a test set of a very different distribution than the training set, it is an effective model.

Once the model was trained, I ran each individual sentence from both training and testing through the model, to render predictions. If the entire feature set is used, this is time-consuming. However, I opted for a "top N" strategy of only retaining the N most prominent features for my testing of the algorithm. With the top N being a small amount, model accuracy suffers. But having the entire feature set is not necessary either. I determined that approximately the top 50,000 features are necessary to be retained to deliver the same accuracy as the full feature set, and shaves off processing time considerably.

My results are shown for the full feature set nonetheless. Naive Bayes drastically outperforms Logistic Regression here. This makes sense, since a Naive Bayes model is a strong choice for a text classification problem. Logistic Regression may be the most popular machine learning classification algorithm in existence, but its efficacy turns out to be not great for this task, or perhaps for text classification in general. Logistic Regression gave me 93% for training and 77% for testing. Naive Bayes gave me 99% for training and 98% for testing. The failures in the LR model are very often confusions of closely related languages (German being mistaken for Dutch, and Spanish being mistaken for Portuguese, are the 2 most common failures in LR). Most of the failures in the Naive Bayes classifier, on the other hand, seem to be from bad training examples. Admittedly, I did not curate "bad" training examples from my data, as I was relying on NLTK's sentence tokenizer to parse out real full sentences for me. It did for the most part, but some notable failures from NB are the strings that the NLTK algorithm parsed as "sentences" but were actually fragments. Ohter times, a string in a different language would show up in an article, and so the ground truth would actually be incorrect here, since ground truths are all labeled based on the article they came from. NB picked up on this too and so delivered a "failure" in these cases.

I wanted to build a language detector as an independent project of personal interest, to see if I could do it, and also to learn more about how to build NLP algorithms and machine learning models. I learned so much about text classification, NLP data processing, and expanded my machine learning skills with this project! I also expanded my skills and comfort level in NLTK, pandas, scikit-learn, and Python programming in general, particularly the file processing aspects of Python. I loved every minute of working on this, and I'm glad to share my passion for NLP and Machine Learning with you! Thanks!
