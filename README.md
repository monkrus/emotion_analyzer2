# emotion_analyzer2
Emotion_analyzer2 is a collection of even more 😏approaches to recognize human emotions.

`pip install requests`

To run :
 `uvicorn faceplus2:app --reload --port 8000`  Face++ Emotion+Pose
 Python code using FastAPI that solely interacts with the Face++ API to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise" **as well as "headpose" and "eyestatus"** (faceplus2.py)


 `uvicorn faceplus1:app --reload --port 8001` Face++ Emotion
 Python code using FastAPI that **solely interacts with the Face++ API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." This code doesn't implement any additional logic beyond what's provided by Face++. (faceplus1.py)


`uvicorn faceazur:app --reload --port 8002` Face++ Emotion+Pose Azur Face Emotion
 Python code using FastAPI that interacts with the **Face++ API and AZURE face API** to detect the emotions "anger," "disgust," "fear," "happiness," "neutral," "sadness," and "surprise." as well as "headpose" and "eyestatus" . Moreover, it compares the results from the both APIs and outputs the most probable result. (faceazur.py)


  