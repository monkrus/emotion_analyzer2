# emotion_analyzer2
Emotion_analyzer2 is a collection of even more 😏approaches to recognize human emotions.

Install:

`pip install requests`

To run :
 `uvicorn app:app --reload --port 8000` (base code)
 Python code using FastAPI that solely interacts with the Face++ API to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." **as well as "headpose" and "eyestatus"** (app.py)


 `uvicorn app1:app --reload --port 8001`
 Python code using FastAPI that **solely interacts with the Face++ API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." This code doesn't implement any additional logic beyond what's provided by Face++. (app1.py)


`uvicorn app:app --reload --port 8002` 
 Python code using FastAPI that interacts with the **Face++ API and AZURE face API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." as well as "headpose" and "eyestatus" (app2.py).Moreover, it compares the results from the both API and outputs the most probable result. (code is progress!)




