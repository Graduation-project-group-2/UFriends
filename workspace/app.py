# pip install Flask==2.1.0
from flask import Flask

app = Flask(__name__)

<<<<<<< HEAD
=======

>>>>>>> ab329c4 (flask2212)
@app.route('/')
def hello_world():
  return 'Hello, World!'

<<<<<<< HEAD
=======

>>>>>>> ab329c4 (flask2212)
@app.route('/user/<userName>')
def hello_user(userName):
  return 'Hello, %s'%(userName)


if __name__ == "__main__":
  app.run()


