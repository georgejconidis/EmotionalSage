## Test file for emotional sage program

import numpy as np
import os
#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

import joblib

from ast import literal_eval

from PIL import Image, ImageDraw, ImageFont

cwd = os.getcwd()


## The pre-selected conversations which the web-app will select from
un_res_indices = [10,20,30,40]
res_indices = [10,20,30,40,50,60,70,80,90,100]

# importing the 'Critical Negative Value' Array
os.chdir('/home/george/Documents/Insight_DS_TO20A/Projects/EmotionalDetection/notebooks')
crictial_neg_val = pd.read_csv('Critical_Neg_Values.csv')
# columns = 	convo_lengths	num_res_convos	num_unres_convos	critical_neg_val	Resolved_frac	Unresolved_frac	Resolved_MIN_Probability
os.chdir(cwd)

# NOTE: These functions were created using the accompanying Jupyetr notebooks for this projects (found in the /notebook folder)

# The Text Preprocesing Function
def text_processing(headline):

    #Generating the list of words in the headline (hastags and other punctuations removed)
    def form_sentence(headline):
        headline_blob = TextBlob(headline)
        return ' '.join(headline_blob.words)
    new_headline = form_sentence(headline)

    #Removing stopwords and words with unusual symbols
    def no_user_alpha(headline):
        headline_list = [ele for ele in headline.split() if ele != 'user']
        clean_tokens = [t for t in headline_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_headline = no_user_alpha(new_headline)

    #Normalizing the words in headlines
    def normalization(headline_list):
        lem = WordNetLemmatizer()
        normalized_headline = []
        for word in headline_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_headline.append(normalized_text)
        return normalized_headline


    return normalization(no_punc_headline)

# The Sentiment Analyzer

def SentimentClassifier(chat_df):
    """
    Classifying the Sentiment of the conversation selected for analysis.

    input: 'text_in': pandas DataFrame where each row contains a tweet for the conversation, already in chronological order

    output: 'text_out' : Is the original pandas DataFrame with the column "Sent_Ints" which is the sentiment cassification of each tweet
    """
    #########################################
    ## Loading the Naive Bayes Classsifier ##
    #########################################
    # The details of this model are:
    # The test data set is balanced to: 0.50097r (random baseline)
    # The OVERALL accuracy of the model is 0.8336
    # This model's accuracy is better than random by 0.33262r
               #   precision    recall  f1-score   support
               #
               # 0       0.81      0.84      0.83       295
               # 1       0.86      0.82      0.84       330
    os.chdir('/home/george/Documents/Insight_DS_TO20A/Projects/EmotionalDetection/data/raw/US_Airline_Sentiment')
    filename = 'NBC_USAirlines_model_Acc83p36.sav'
    pipeline = joblib.load(filename)
    text_temp = chat_df['Text']
    chat_df['Sentement_Ints'] = pipeline(text_temp)
    return chat_df


os.chdir(cwd)

# The Plot Generator Function
def ES_Sentiment_Plot(chat_df):
    """
    This function generates the sentiment plot given a conversation.
    """
    index_agent = chat_df.index[chat_df['speaker'] == 'CS_Agent' ].tolist()
    index_customer = chat_df.index[chat_df['speaker'] == 'Customer' ].tolist()

    # fig, ax = plt.subplots()
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(index_agent,chat_df['emotions'][index_agent],label='agent',lw=7,color='lightgreen')
    plt.plot(index_customer,chat_df['emotions'][index_customer],label='customer', lw=7,color='lightblue')

    #plt.axvline(x=37,linewidth=10,color='r',label="Intervention")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,prop={'size': 30} )
    plt.xlabel('Tweet Number', size=40)
    plt.ylabel('Emotion',size=40)
    plt.xticks(fontsize=25)
    plt.yticks([0,1],labels=['Positive','Negative'],fontsize=40)

    #plt.text(10.1,0,'Intervention',rotation=90)
    # Changing the y-labels to "Sad" (0) and "Happy" (1)
    # labels = [item.get_text() for item in ax.get_yticklabels()]
    # labels[0] = 'Sad'
    # labels[1] = 'Happy'
    plt.xlim(0,10)
    plt.savefig("Preliminary_AgentCustomer_Emotional_Output.png", bbox_inches='tight', dpi=100)
    plt.plot()


# The Animation Generator Function

def ES_animation_generator():
    """
    This function generates the senitment animation plot given a conversation.
    """
#sns.palplot(sns.color_palette("husl", 8)) #

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    t = np.linspace(1, len(CSChat_Emotions), len(CSChat_Emotions))
    x = CSChat_Emotions["Customer_Emotions"]
    y = CSChat_Emotions["Agent_Emotions"]
    # print(t.shape)
    # print(x.shape)
    # print(y.shape)

    ax1.set_ylabel(u'Customer')
    ax1.yaxis.set_label_position("right")
    ax1.set_xlim(0, max(CSChat_Emotions.index))
    ax1.set_ylim(-0.05, 1.05)
    plt.setp(ax1.get_xticklabels(),visible=False)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(labels=['Postive', 'Negative'])
    ax1.set_title('Real-time Emotions: Customer Service Text Conversation', size=15)

    ax2.set_xlabel('Text Message Number')
    ax2.set_ylabel(u'CS Agent')
    ax2.yaxis.set_label_position("right")
    ax2.set_xlim(0, max(CSChat_Emotions.index))
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(labels=['Postive', 'Negative'])

    lines = []
    for i in range(1,len(t)):
    #     print(i)
        head = i - 1
        head_slice = (t > t[i] - 1.0) & (t < t[i])
        line1,  = ax1.plot(t[:i], x[:i], color='lightblue')
        line1a, = ax1.plot(t[head_slice], x[head_slice], color='red', linewidth=2)
        line1e, = ax1.plot(t[head], x[head], color='red', marker='o', markeredgecolor='r')
        line2,  = ax2.plot(t[:i], y[:i], color='lightgreen')
        line2a, = ax2.plot(t[head_slice], y[head_slice], color='red', linewidth=2)
        line2e, = ax2.plot(t[head], y[head], color='red', marker='o', markeredgecolor='r')
        lines.append([line1,line1a,line1e,line2,line2a,line2e])


    # Build the animation using ArtistAnimation function

    ani = animation.ArtistAnimation(fig,lines,interval=125,blit=True)
    ani.save('animation.gif', writer='imagemagick', fps=10)

def PlottingTweetsonImage(df_chat):
    """
    Plotting the text of the Tweet on an Image.
    Highlighting ANY tweet which suprass the Critical Negativity Value
    """
    # Gathering the text to be plotted
    lines = literal_eval(df_chat['Tweets_Sorted']) # Tweets
    Sent_list = literal_eval(df_chat['Emotions_Sorted']) # Sent (words)
    Sent_Ints = literal_eval(df_chat['Emotions_Sorted_Ints'])
    # Determining the Critical Value
    Critical_Values = crictial_neg_val

    # Determine where the conversation surpasses the critical negativity threshold
    Neg_Value_temp = []
    for i in range(len(Sent_Ints)):
        Neg_Value_temp.append(Sent_Ints[i]*1.0/(i+1))

        # +1 b/c list start their index count at 0
    print("The Neg Value Array", Neg_Value_temp)
    i_bad = -1 # defnining the index of critical neg. value such that it will not be plotted if the conversation never hits the critical neg. value
    for i in range(len(Neg_Value_temp)):
        if Neg_Value_temp[i] > Critical_Value:
            i_bad = i
            print("i-bad value is =", i_bad)
            break

    img = Image.new('RGB', (1800, 400), color = (240, 255, 240))
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 26)
    d = ImageDraw.Draw(img)
    for i in range(len(lines)):
        message = lines[i]
        sent_temp = Sent_list[i]
        if Neg_Value_temp[i] >= Critical_Value and i >1:
            d.text((10,10+36*i),"*"+sent_temp+"--"+message+"*", fill=(255,20,20), font=font)
        else: # Neg_Value_temp[i] < Critical_Value:
            d.text((10,10+36*i),sent_temp+"--"+message, fill=(100,20,255), font=font)
    # file_name = "Text_%" %(i)
    file_name = "ConvoText.png"
    # print(file_name)
    os.chdir('/home/george/Documents/Insight_DS_TO20A/Projects/EmotionalDetection/projectname/static')
    img.save(file_name)
    os.chdir(cwd)
    return file_name
