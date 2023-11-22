pip install wordcloud
import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

MainTab, EDA, preprocess, test, results = st.tabs(["Main", "Explorative Data Analysis", "Data Preprocessing", "Testing", "Results"])

with MainTab:

    # Check if NLTK stopwords are already downloaded
    try:
        # Attempt to find the stopwords dataset
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        # If not found, download the stopwords dataset
        nltk.download('stopwords')
    
    # Define preprocessing functions
    def remove_stopwords(text):
        text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
        return text
    
    def remove_url(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)
    
    def remove_punct(text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)
    
    def remove_html(text):
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)
    
    def remove_username(text):
        return re.sub('@[^\s]+', '', text)
    
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def decontraction(text):
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
        
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text  
    
    def separate_alphanumeric(text):
        words = text
        words = re.findall(r"[^\W\d_]+|\d+", words)
        return " ".join(words)
    
    def cont_rep_char(text):
        tchr = text.group(0) 
        
        if len(tchr) > 1:
            return tchr[0:2] 
    
    def unique_char(rep, text):
        substitute = re.sub(r'(\w)\1+', rep, text)
        return substitute
    
    def char(text):
        substitute = re.sub(r'[^a-zA-Z]',' ',text)
        return substitute
    
    # Function to preprocess input text
    def preprocess_input_text(input_text):
        text = input_text
        text = remove_username(text)
        text = remove_url(text)
        text = remove_emoji(text)
        text = decontraction(text)
        text = separate_alphanumeric(text)
        text = unique_char(cont_rep_char, text)
        text = char(text)
        text = text.lower()
        text = remove_stopwords(text)
        return text

    # Function to generate a word cloud
    def generate_wordcloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    # Function to get the top N occurring words
    def get_top_words(text, n=10):
        words = text.split()
        word_counts = Counter(words)
        top_words = word_counts.most_common(n)
        return top_words
    
    # Load your pre-trained models (model1 and model2)
    model1 = joblib.load("bnb_smote.pkl")  # Replace with your model file path
    model2 = joblib.load("LinearSVC_smote.pkl")
    tfidf_vectorizer2 = joblib.load("tfidf_vectorizer.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizerbnb.pkl")
    
    # Create a Streamlit app
    st.title("Sentiment Analysis Application For Flight Reviews")
    
    st.markdown("Created by 📈 Ng Jia Jun JD")
    
    # Create a form to encapsulate the input fields
    with st.form("text_prediction_form"):
        # Create a selectbox to allow the user to choose the model
        prediction_type = st.selectbox("Select a prediction type", ["Text Prediction", "File Prediction"])
        
        # Create a text input field
        user_input = st.text_area("Enter Text Prediction Here:", "")
    
        # Create a selectbox to allow the user to choose the model
        selected_model = st.selectbox("Select a Model", ["BernoulliNB", "LinearSVC"])
    
        #provide uplaod file platform
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        # Create a button to make predictions
        prediction_button = st.form_submit_button("Make Prediction")
    
        
    
    
    # Check if the form is submitted
    if prediction_button:
        if prediction_type == "Text Prediction":
            if not user_input:
                st.error("Please enter text for prediction.")
            else:
                if selected_model == "BernoulliNB":
                    # Preprocess the user input for TF-IDF and BoW features
                    preprocessed_input = preprocess_input_text(user_input)
    
                    # Transform the preprocessed input using the same TF-IDF vectorizer
                    tfidf_input = tfidf_vectorizer.transform([preprocessed_input])
    
                    # Make predictions using model1
                    prediction = model1.predict(tfidf_input)
    
                    # Display the preprocessed input
                    st.write(f"Preprocessed text: {preprocessed_input}")
    
                    # Display the prediction result
                    st.write(f"Prediction: {prediction}")
                else:
                    # Preprocess the user input for TF-IDF and BoW features
                    preprocessed_input = preprocess_input_text(user_input)
    
                    # Transform the preprocessed input using the same TF-IDF vectorizer
                    tfidf_input = tfidf_vectorizer2.transform([preprocessed_input])
    
                    # Make predictions using model2
                    prediction = model2.predict(tfidf_input)
    
                    # Display the preprocessed input
                    st.write(f"Preprocessed text: {preprocessed_input}")
    
                    # Display the prediction result
                    st.write(f"Prediction: {prediction}")
        else:
            if not uploaded_file:
                st.error("Please upload a CSV file for prediction.")
            else:
                if selected_model == "BernoulliNB":
                    data = pd.read_csv(uploaded_file, encoding='latin1')
    
                    # Apply preprocessing to the 'text' column using .apply()
                    data['cleaned_data'] = data['text'].apply(preprocess_input_text)
    
                    tfidf_features = tfidf_vectorizer.transform(data['cleaned_data'])
    
                    # Make predictions using model1
                    predictions = model1.predict(tfidf_features)
    
                    prediction_counts = pd.Series(predictions).value_counts()
                    plt.figure(figsize=(8, 6))
                    plt.bar(prediction_counts.index, prediction_counts.values, tick_label=['Neutral', 'Positive', 'Negative'])
                    plt.xlabel("Sentiment")
                    plt.ylabel("Count")
                    plt.title("Sentiment Analysis Results")
                    plt.ylim(0, 15000)  # Set the Y-axis limit to 1000 per inch
                    st.pyplot(plt)

                    cleaned_text_data = data['cleaned_data'].str.cat(sep=' ')

                    # Display word cloud
                    st.subheader("Word Cloud")
                    generate_wordcloud(cleaned_text_data)
                    
                    # Display top 10 occurring words
                    st.subheader("Top 10 Occurring Words")
                    top_words = get_top_words(cleaned_text_data, n=10)
                    st.write(pd.DataFrame(top_words, columns=['Word', 'Count']))
                else:
                    data = pd.read_csv(uploaded_file, encoding='latin1')
    
                    # Apply preprocessing to the 'text' column using .apply()
                    data['cleaned_data'] = data['text'].apply(preprocess_input_text)
    
                    tfidf_features = tfidf_vectorizer2.transform(data['cleaned_data'])
    
                    # Make predictions using model2
                    predictions = model2.predict(tfidf_features)
    
                    prediction_counts = pd.Series(predictions).value_counts()
                    plt.figure(figsize=(8, 6))
                    plt.bar(prediction_counts.index, prediction_counts.values, tick_label=['Neutral', 'Positive', 'Negative'])
                    plt.xlabel("Sentiment")
                    plt.ylabel("Count")
                    plt.title("Sentiment Analysis Results")
                    plt.ylim(0, 15000)  # Set the Y-axis limit to 1000 per inch
                    st.pyplot(plt)

                    cleaned_text_data = data['cleaned_data'].str.cat(sep=' ')

                    # Display word cloud
                    st.subheader("Word Cloud")
                    generate_wordcloud(cleaned_text_data)
                    
                    # Display top 10 occurring words
                    st.subheader("Top 10 Occurring Words")
                    top_words = get_top_words(cleaned_text_data, n=10)
                    st.write(pd.DataFrame(top_words, columns=['Word', 'Count']))

with EDA:
    st.title("Sentiment Analysis on Airline Reviews: A Comparison Study on Machine Learning Models")
    
    st.markdown("by Ng Jia Jun JD")

    st.header('Brief look at the dataset used (Tweets.csv)')
    st.image('./snapshots/tweets.jpg')
    st.write('According to the owner (Crowdflower), the dataset used for this project was crawled during February of 2015 on Twitter (now known as X) and the dataset has a total of 14,640 sample data that contains customer’s opinion regarding the United States Airline that is divided into 3 sentiment classes such as positive, neutral and negative.')

    st.header('Dataset Information / Columns')
    st.image('./snapshots/data info.jpg')
    st.write('Before we begin with the preprocessing of the raw text data, we will perform an Explorative Data Analysis on the dataset. Firstly, we will check the columns in the dataset. There are a total of 15 columns. The column that we are mainly focusing on for this project is ‘airline_sentiment’ and ‘text’.')

    st.header('First and last tweet created')
    st.image('./snapshots/firstlasttweetcreated.jpg')
    st.write('The dataset consists of tweets ranging from the 16th of February 2015 to 25 February 2015. This means that this dataset consists of 9 days of data.')

    st.header('Check for null values & Remove it')
    st.image('./snapshots/nullvalues.jpg')
    st.write('It can be seen that column ‘airline_sentiment_gold’, ‘negativereason_gold’ and ‘tweet_coord’ all have null values upwards to 90%. Therefore, we will drop these columns as they do not provide any impactful information.')

    st.header('Distribution of Sentiments')
    st.image('./snapshots/sentiment distribution.jpg')
    st.write('We can see that most of the customer reviews regarding the airline are mostly negative totalling at 9178 which takes up 62.7% of the total number of sample data. As for the neutral and positive sentiment, it is at 3099 (21.2%)  and 2363 (16.1%) respectively. Moreover, we can clearly see that the dataset is imbalanced and will require oversampling during model training and testing.')

    st.header('Distribution of Airline types')
    st.image('./snapshots/airline types distribution.jpg')
    st.write('According to the bar chart, most of the tweets are conveyed towards United standing at 26.1%. The second highest is US Airways at 19.9%. The third highest is American at 18.8% and the remaining Southwest, Delta and Virgin America are at 16.5%, 15.2% and 3.4% respectively.')

    st.header('Distribution of sentiment for each Airline types')
    st.image('./snapshots/sentiment for each airline distribution.jpg')
    st.write('According to the bar chart shown, it can be seen that all of the airline types received more negative reviews than positive and neutral.')

    st.header('Overall distribution for negative reasons')
    st.image('./snapshots/overall distribution for negative reasons.jpg')
    st.write('From the donut pie chart, we can clearly see that most of the negative reasons are related to customer service issues at 31.7%. The second highest negative reasons are due to late flight at 18.1%.')

    st.header('Wordcloud for Positive Reasons')
    st.image('./snapshots/positivewc.png')

    st.header('Wordcloud for Neutral Reasons')
    st.image('./snapshots/neutralwc.png')

    st.header('Wordcloud for Negative Reasons')
    st.image('./snapshots/negativewc.png')
    st.write('The 3 Wordcloud shown above are for Positive, Neutral and Negative sentiments in the dataset. Using wordcloud, we can get a sense of the data before diving into more detailed sentiment scoring and analysis. The more prominent the word, the more the word occurs.')
    

with preprocess:
    st.title("Data Preprocessing of Raw Text Data (Tweets.csv)")
    st.write('Before we begin with testing and training the models, we will be preprocessing the dataset. Preprocessing refers to the text cleaning and transformation steps applied to raw text data before it is fed into a machine learning model to analyze the sentiment expressed in the text. (Kosaka, 2020). \nThe preprocessing steps that were applied in this assignments are as follows: \n 1. Removing url, punctuation, html, @username, emoji \n 2. Decontration (Expand words to full form like “can’t = “cannot”) \n 3. Stop word removal (Remove common words that do not have significant sentiment information) \n 4. Convert text to lowercase')
    
    st.header('Remove stop words, URL, punctuation, html, username & emojis')
    st.image('./snapshots/pp1.jpg')

    st.header('Decontraction of text')
    st.image('./snapshots/pp2.jpg')

    st.header('Separate alphanumerics')
    st.image('./snapshots/pp3.jpg')

    st.header('Apply all preprocessing functions onto the raw text data')
    st.image('./snapshots/pp4.jpg')

    st.header('Before & After comparison of Data Preprocessing')
    st.image('./snapshots/pp5.jpg')

with test:
    st.title('Training and Testing code for BernoulliNB and LinearSVC')
    st.write('Before we begin searching for the best and optimized model, we will first perform a baseline performance on each model using the default hyperparameter for the model and TF-IDF. After that, we will be fine tuning the hyperparameter for TF-IDF and each model using Grid Search Cross Validation with 5 folds. After fine tuning the hyperparameter for TF-IDF and each model and have obtained the best hyperparameter for each model, we will be applying Synthetic Minority Oversampling Technique (SMOTE) on both the best models to adjust the distribution of the classes so that it can be trained in a balanced data. From there on, we will determine which will be the best model suitable for this project.')

    st.header('Baseline Model')
    st.subheader('Baseline model training and testing for BernoulliNB')
    st.image('./snapshots/bnbtest1.jpg')
    st.subheader('Baseline model training and testing for LinearSVC')
    st.image('./snapshots/lsvctest1.jpg')

    st.header('Baseline Model + Hyperparameter Tuning using GridSearchCV')
    st.subheader('Baseline BernoulliNB with Hyperparameter Tuning using GridSearchCV')
    st.image('./snapshots/bnbtest2.jpg')
    st.subheader('Baseline LinearSVC with Hyperparameter Tuning using GridSearchCV')
    st.image('./snapshots/lsvctest2.jpg')

    # Create a DataFrame with 2 rows and 6 columns
    testhyperparameter = {'Hypeparameter': ['BernoulliNB (alpha)', 'LinearSVC (C)', 'max_features', 'max_df', 'ngram_range'],
            'Tested Values': ['[0.1, 0.5, 1.0]', '[0.1, 1, 10]','[0.25, 0.5, 0.75]', '[2500, 5000, 10000]','[(1,1), (1,2), (1,3)]' ]}
    
    table1 = pd.DataFrame(testhyperparameter)
    
    # Display the table in Streamlit
    st.table(table1)

    st.header('Best Model + Oversampling with SMOTE')
    st.subheader('Best BernoulliNB with Oversampling with SMOTE')
    st.image('./snapshots/bnbtest3.jpg')
    st.subheader('Best LinearSVC with Oversampling with SMOTE')
    st.image('./snapshots/lsvctest3.jpg')

    # Create a DataFrame with 2 rows and 6 columns
    besthyperparameter = {'Models': ['Best Value (Model Hyperparameter)', 'Best Value (max_features)', 'Best Value (max_df)', 'Best Value (ngram_range)'],
                          'BernoulliNB': ['alpha: 1.0', '5000','0.5', '1,3'],
                          'LinearSVC': ['C: 0.1', '2500', '0.25', '1,3']
                         }
    
    table2 = pd.DataFrame(besthyperparameter)
    
    # Display the table in Streamlit
    st.table(table2)

with results:
    st.header('Baseline performance of BernoulliNB & LinearSVC')
    st.image('./snapshots/baseline.jpg')
    
    st.header('Baseline performance of BernoulliNB & LinearSVC + Hyperparameter Tuning using GridSearchCV')
    st.image('./snapshots/baseline tune.jpg')

    st.header('Best BernoulliNB & LinearSVC performance + Oversampling with SMOTE')
    st.image('./snapshots/best model smote.jpg')

    
