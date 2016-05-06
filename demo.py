from flask import Flask, render_template, request, jsonify, make_response, json
import numpy as np
import Stackoverflow as so

# Initialize the Flask application
app = Flask(__name__)

#Secret Key for Sessions.
app.secret_key = '\xd4}C\xa4\x03b\n\xfdo\xbc\xab\xa4\x01\x91JJ\xfe-\x8d\xc7\x04\xe0[('

@app.route("/")
def hello():
	return "Hello World!"

@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/Refresh")
def Refresh():
    global refresh
    refresh = True
    print refresh
    return render_template('index.html')

# Route that will process the AJAX request, sum up two
# integer numbers (defaulted to zero) and return the
# result as a proper JSON response (Content-Type, etc.)
@app.route('/GetAnswer',methods=['GET','POST'])
def GetAnswer():
    userid = request.args.get('userid')
    title = request.args.get('title')
    body = request.args.get('body')

    # print userid
    # print title
    # print body

    full_question = title + " " + body

    tags = so.demo(full_question, userid)
    temp_str = ""
    for t in tags:
        temp_str += t + "       "
    print temp_str

    {'html':'<span>'+temp_str+'</span>'}

    # validate the received values
    # if userid and title and body:
    #     return json.dumps({'html':'<span>All fields good !!</span>'})
    # else:
    #     return json.dumps({'html':'<span>Enter the required fields</span>'})

   
    # b = [{'answer' : temp_str, 'next_answer' : ['test', 'test', 'test']}]
    

    b = []
    for t in tags:
        b.append({'tag' : t})

    print "json"
    print b
    print "json"

    return jsonify(result=b)

if __name__ == '__main__':
    global refresh
    global mat #matrix
    global total
    global prev_question
   
    refresh = True
    app.run(
    	port=5001,
        debug=True
    )

