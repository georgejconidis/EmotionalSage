from flask import Flask, render_template, request, Response

## George's custom imports
import os
import random
#Data Analysis
import numpy as np
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

# Converts a string list to a list with the command: literal_eval(df['column'].iloc[i])

from ast import literal_eval

# Plotting the text on an image
from PIL import Image, ImageDraw, ImageFont


cwd = os.getcwd()

## Now importing the Emotional Sage modules
os.chdir('/home/george/Documents/Insight_DS_TO20A/Projects/EmotionalDetection/projectname/projectname')
from projectname.EmotionalSage import text_processing, SentimentClassifier, ES_Sentiment_Plot, ES_animation_generator, PlottingTweetsonImage

html_script = 'product-page.html'

### Importing DataFrame with conversations
os.chdir('/home/george/Documents/Insight_DS_TO20A/Projects/EmotionalDetection/notebooks')
Res = pd.read_csv('SUB_Conv_Resolved_len5more.csv')
UNRes = pd.read_csv('SUB_Conv_UNResolved_len5more.csv')
os.chdir(cwd)

## The pre-selected conversations which the web-app will select from
unres_indices = [10,20,30,40]
res_indices = [5,15,25,35] # 35, are good :)

########################################################################

# Create the application object
app = Flask(__name__)
# GoogleMaps(app,key=apikey)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
    return render_template(html_script)  # render a template

@app.route('/output')
def recommendation_output():
    # print('Test')
   	# Pull input
    # address_input = "Stuff"#request.args.get('address') + ', Boston, MA'
    #property_type = request.args.get('property type')
    neighborhood_temp_input = request.args.get('neighborhood')
    #bed_input = float(request.args.get('bed'))
    #bath_input = float(request.args.get('bath'))
    #size_input = float(request.args.get('size'))
    #year_input = int(request.args.get('year built'))

    # Initializing the variables
    conv_type = '' # This is the type of conversation desired by the user (resolved/unresolved)
    conversation_figure = '' # This is the Text in the tweets
    Sentiment_gif = '' # This is the animation of the conversation
    Sentiment_Figure ='' # This is a still image
    Critical_Negativity_Value = 0
    random_index_selcted = -1
    # age_input = current_year - year_input

    # print(neighborhood_temp_input)
    # if neighborhood_temp_input == "Resolved":
    #     conv_type = "Resolved Conversation"
    #     conversation_figure = 'Preliminary_AgentCustomer_Emotional_Output.png'
    #     Sentiment_gif ='animation1.gif'
    #     Sentiment_Figure ='Resolved_Sentiment.png' # This is a still image of the gif above
    #     Critical_Negativity_Value = "0.415"
    # elif neighborhood_temp_input == "Unresolved":
    #     conv_type = "Unresolved Conversation"
    #     conversation_figure = 'customer_service_lasercycleusa.jpg'
    #     Sentiment_gif ='animation1.gif'
    #     Sentiment_Figure ='Unresolved_Sentiment.png' # This is a still image of the gif above
    #     Critical_Negativity_Value = "0.415"


    # property_latitude,property_longitude = get_coor(address_input)
    # dist_to_poi = np.array([calculate_avg_dist_to_POI(property_longitude,property_latitude,
    #                                         shop) for shop in shop_category])
    #
    # crime_rate = calculate_crime_density(property_longitude,property_latitude,
    #                                  xx_grid,yy_grid,crime_grid)
    #
    # neighborhood_avg_sold_price = df_neighborhood_avg['last avg sold price'].loc[neighborhood_temp_input]
    # neighborhood_avg_time_to_sell = df_neighborhood_avg['last avg time to sell'].loc[neighborhood_temp_input]
    # print('%s last average sold price: %.2f' % (neighborhood_temp_input, neighborhood_avg_sold_price))
    # print('%s last average time to sell: %.2f' % (neighborhood_temp_input, neighborhood_avg_time_to_sell))

    # property_type_input = one_hot_encoding(property_type, one_hot_type, 'PROPERTY TYPE')
    # print('property_type_input: ',property_type_input)random_index_selected
    #
    # regr_sold = pickle.load(open('./pickled_model/RF_regr_sold_price_all.pkl', 'rb'))
    # regr_tts = pickle.load(open('./pickled_model/GradBoost_regr_tts_all.pkl', 'rb'))
# Case if empty
## GC: This is calling on the model predictions
    if neighborhood_temp_input == "":
            return render_template(html_script,
                                   my_form_result="Empty")
    else:
        my_form_result="NotEmpty"
        if neighborhood_temp_input == "Resolved":
            conv_type = "Resolved Conversation"
            # Selecting the converstion to be shown (randomly)
            random_index_selected = random.choice(res_indices)
            print("The Resolved Index Choosen =",random_index_selected)
            # Gathering the DataFrame Slice of for the selected Conversation
            df_conv_all = Res.iloc[random_index_selected]
            # Generating the Conversation Text image
            conversation_figure = ''
            conversation_figure = PlottingTweetsonImage(df_conv_all)
            print(conversation_figure)
            # conversation_figure = 'ConvoText1.png'
            Health_image = "https://i1.rgstatic.net/ii/profile.image/279449833623559-1443637440288_Q512/George_Conidis.jpg"

            Sentiment_gif ='animation1.gif'
            # Sentiment_Figure ='Resolved_Sentiment.png' # This is a still image of the gif above
            Critical_Negativity_Value = "0.415"
        elif neighborhood_temp_input == "Unresolved":
            conv_type = "Unresolved Conversation"
            # Selecting the converstion to be shown (randomly)
            random_index_selected = random.choice(unres_indices)
            print("The Unresolved Index Choosen =",random_index_selected)
            # Gathering the DataFrame Slice of for the selected Conversation
            df_conv_all = UNRes.iloc[random_index_selected]
            # Generating the Conversation Text image
            conversation_figure = PlottingTweetsonImage(df_conv_all)
            print(conversation_figure)
            Sentiment_gif ='animation1.gif'
            # Sentiment_Figure ='Unresolved_Sentiment.png' # This is a still image of the gif above
            Critical_Negativity_Value = "0.415"
            Health_image = "https://i1.rgstatic.net/ii/profile.image/279449833623559-1443637440288_Q512/George_Conidis.jpg"

        return render_template(html_script, conv_type = conv_type, Critical_Negativity_Value = Critical_Negativity_Value, Sentiment_gif = Sentiment_gif, conversation_figure = conversation_figure, my_form_result=my_form_result, Health_image = Health_image)



# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True,port=8000)  # will run locally http://127.0.0.1:5000/
