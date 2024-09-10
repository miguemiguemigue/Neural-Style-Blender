from flask import Flask

app = Flask(__name__)

# Define the endpoint for the root URL
@app.route('/')
def hello_world():
    return 'Hola Mundo!'

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
