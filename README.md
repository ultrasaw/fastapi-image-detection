## venv
python3 -m venv venv_img

## pip
pip freeze > requirements.txt
pip install -r requirements.txt

## start the application
uvicorn main:app --reload