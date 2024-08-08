from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests
import os

app = FastAPI()

# Face++ API Configuration
FACEPP_API_KEY = "xxx"
FACEPP_API_SECRET = "xxx"
FACEPP_API_ENDPOINT = "https://api-us.faceplusplus.com/facepp/v3/detect"

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/detect_attributes")
async def detect_attributes(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        files = {
            'image_file': contents
        }
        data = {
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'return_attributes': 'headpose,eyestatus,emotion'
        }
        response = requests.post(FACEPP_API_ENDPOINT, files=files, data=data)
        response.raise_for_status()
        
        faces = response.json().get('faces', [])
        if faces:
            face_attributes = faces[0]['attributes']
            head_pose = face_attributes['headpose']
            eyestatus = face_attributes['eyestatus']
            emotion = face_attributes['emotion']
            return JSONResponse({
                "headPose": head_pose,
                "eyeStatus": eyestatus,
                "emotion": emotion
            })
        return JSONResponse({"error": "No face detected"}, status_code=400)
    except requests.RequestException as e:
        return JSONResponse({"error": f"Request error: {str(e)}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Internal Server Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
