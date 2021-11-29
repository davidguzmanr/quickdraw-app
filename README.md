# quickdraw-app
A simple implementation of the game [Quick, Draw!](https://quickdraw.withgoogle.com/) from Google.

First, create a virtual environment:

```
virtualenv quickdraw-venv
```

Then activate it:
```
source quickdraw-venv/bin/activate
```

Then clone this repository and install the requirements:
```
git clone https://github.com/davidguzmanr/quickdraw-app.git
cd quickdraw-app
pip install -r requirements.txt
```

Then run the app:
```
streamlit run app.py
```