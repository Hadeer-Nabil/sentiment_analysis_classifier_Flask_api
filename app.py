from flask import Flask
from flask_restful import Resource, Api
from flask import request
import pickle
app = Flask(__name__)
api = Api(app)
import re

# load the model from disk
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))


def clean_text(text):
    """
        clean data from special characters and html tags
    """
    
    text = text.strip().lower()
    
    # remove tags 
    text = re.sub(r'<.*?>', '', text)
            
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    text = ''.join((filter(lambda x: x not in filters.split(), text)))

    return text

# Loading features

tf = pickle.load(open("tfidf.pickle", 'rb'))


class predict(Resource):

    def get(self):
        comment = request.args.get('comment')
        comment_score = loaded_model.predict(tf.transform([comment])).tolist()[0]
        return 'Negative' if comment_score == 0 else 'Positive' 



api.add_resource(predict, '/predict')
if __name__ == '__main__':

    app.run(debug=True)
