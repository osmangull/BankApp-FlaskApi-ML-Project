import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelM4A.pkl', 'rb'))


@app.route('/m4a_Api',methods=['POST'])
def m4a_Api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    if output>=1.0 and output<1.5:
        return jsonify('M4ABank Mesajı:Kredi Verilebilir')
    elif output>=1.5:
       return jsonify('M4ABank Mesajı: Kredi Verilemez')
    else:
       return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)
