from flask import Flask, render_template, request


import os
import numpy
cwd = os.getcwd()
# os.chdir('/home/george/Documents/Insight_DS_TO20A/Projects/EmotionalDetection/projectname/projectname')
# from projectname.EmotionalSage import test_round
# os.chdir(cwd)

# Create the application object
app = Flask(__name__)

# @app.route("/")
# def home_page():
# 	return render_template('index.html')

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
	return render_template('blog-simple.html')  # render a template

# @app.route('/output',methods=["GET","POST"])
# def recommendation_output():
# 	some_number = "Stuff"
# 	return render_template("index0.html",
# 						my_number=some_number)

#
# @app.route('/output')#,methods=["GET","POST"])
# def recommendation_output():
# #
# 	# Pull input
# 	some_input =request.args.get('rating')
# 	# convo_selected = request.args.get('rating')
# 	# Case if empty
# 	# if some_input =="'":
# 	# 	return render_template("index.html",
# 	#                       	my_input = some_input,
# 	#                       	my_form_result="Empty")
# 	# else:
# 	some_output="yeay!"
# 	some_number=3
# 	some_image="animation1.gif"
# 	return render_template("blog-simple.html",
#                   	my_input=some_input,
#                   	my_output=some_output,
#                   	my_number=some_number,
#                   	my_img_name=some_image,
#                   	my_form_result="NotEmpty")

# @app.route('/output',methods=["GET","POST"])
# def recommendation_output():
# #
# 	# Pull input
# 	# some_number = request.args.get('rating')
# 	some_number = "Stuff"
# 	# convo_selected = request.args.get('rating')
# 	# Case if empty
#
# 	return render_template("index.html",
# 						my_number=some_number)
#
# 	# return render_template("index.html",
# 	# 	                  	my_input=some_input,
# 	# 	                  	my_output=some_output,
# 	# 	                  	my_number=some_number,
# 	# 	                  	my_img_name=some_image,
# 	# 	                  	my_form_result="NotEmpty")
#
#
# # To view the contents of the CS Agnet and Customer conversation
#
# @app.route("/submitted", methods=['POST'])
# def hello():
#    myvariable = request.form.get("rating")
#    return print(myvariable)


# @app.route('/home/george/Documents/Insight_DS_TO20A/Projects/EmotionalDetection/data/raw')
# def content():
# 	text = open('real_chat.txt', 'r+')
# 	content = text.read()
# 	text.close()
# 	print(text)
# 	return render_template('content.html', text=content)

# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
