from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import os

app = FastAPI()

# Load environment variables from .env file
load_dotenv()
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")
FACEPP_API_ENDPOINT = os.getenv("FACEPP_API_ENDPOINT")

# Ensure that the API key, secret, and endpoint are set
if not FACEPP_API_KEY or not FACEPP_API_SECRET or not FACEPP_API_ENDPOINT:
    raise ValueError("One or more environment variables for Face++ API are not set.")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main_page():
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
            'return_attributes': 'emotion'
        }
    
        response = requests.post(FACEPP_API_ENDPOINT, files=files, data=data)
        
        # Print response details for debugging (optional)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)

        response.raise_for_status()
        
        faces = response.json().get('faces', [])
        if faces:
            face_attributes = faces[0]['attributes']
            emotion = face_attributes['emotion']
            
            return JSONResponse({
                "emotion": emotion
            })
        return JSONResponse({"error": "No face detected"}, status_code=400)
    except requests.RequestException as e:
        return JSONResponse({"error": f"Request error: {str(e)}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Internal Server Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("faceplus1:app", host="0.0.0.0", port=8001, reload=True)
