import os
from flask import Flask, request, render_template
from model_loader import predict_spam

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = ''
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if message:
            result = predict_spam(message)
    return render_template('index.html', result=result, message=message)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
