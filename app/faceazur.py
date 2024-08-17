from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import os
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET")
FACEPP_API_ENDPOINT = os.getenv("FACEPP_API_ENDPOINT")
AZURE_FACE_API_KEY = os.getenv("AZURE_FACE_API_KEY")
AZURE_FACE_API_ENDPOINT = os.getenv("AZURE_FACE_API_ENDPOINT") + "/face/v1.0/detect"

# Ensure that all necessary API keys and endpoints are set
if not FACEPP_API_KEY or not FACEPP_API_SECRET or not FACEPP_API_ENDPOINT:
    raise ValueError("One or more environment variables for Face++ API are not set.")
if not AZURE_FACE_API_KEY or not AZURE_FACE_API_ENDPOINT:
    raise ValueError("One or more environment variables for Azure Face API are not set.")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Function to get head pose and eye status from Face++ API
async def get_headpose_eyestatus_facepp(contents):
    files = {'image_file': contents}
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'return_attributes': 'headpose,eyestatus'
    }
    response = requests.post(FACEPP_API_ENDPOINT, files=files, data=data)
    
    # Log response details for debugging
    logging.info(f"Face++ HeadPose & EyeStatus Response Status Code: {response.status_code}")
    logging.info(f"Face++ HeadPose & EyeStatus Response Text: {response.text}")

    response.raise_for_status()
    
    faces = response.json().get('faces', [])
    if faces:
        headpose = faces[0]['attributes']['headpose']
        eyestatus = faces[0]['attributes']['eyestatus']
        return headpose, eyestatus
    return None, None

# Function to get emotion from Face++ API
async def get_emotion_facepp(contents):
    files = {'image_file': contents}
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'return_attributes': 'emotion'
    }
    response = requests.post(FACEPP_API_ENDPOINT, files=files, data=data)
    
    # Log response details for debugging
    logging.info(f"Face++ Emotion Response Status Code: {response.status_code}")
    logging.info(f"Face++ Emotion Response Text: {response.text}")

    response.raise_for_status()
    
    faces = response.json().get('faces', [])
    if faces:
        emotion = faces[0]['attributes']['emotion']
        dominant_emotion = max(emotion, key=emotion.get)
        return dominant_emotion, emotion
    return None, {}

# Function to get emotion from Azure Face API
async def get_emotion_azure(contents):
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_FACE_API_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {'returnFaceAttributes': 'emotion'}
    
    response = requests.post(AZURE_FACE_API_ENDPOINT, headers=headers, params=params, data=contents)
    
    # Log response details for debugging
    logging.info(f"Azure Response Status Code: {response.status_code}")
    logging.info(f"Azure Response Text: {response.text}")

    response.raise_for_status()
    
    faces = response.json()
    if faces:
        emotions = faces[0]['faceAttributes']['emotion']
        dominant_emotion = max(emotions, key=emotions.get)
        return dominant_emotion, emotions
    return None, {}

# Helper function to calculate consensus from two emotion dictionaries
def calculate_consensus(emotion1, emotion2):
    consensus_emotion = {}
    for emotion in set(emotion1.keys()).intersection(emotion2.keys()):
        consensus_emotion[emotion] = (emotion1[emotion] + emotion2[emotion]) / 2
    if consensus_emotion:
        dominant_emotion = max(consensus_emotion, key=consensus_emotion.get)
        return dominant_emotion, consensus_emotion
    return None, {}

@app.post("/detect_emotions_consensus")
async def detect_emotions_consensus(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Get head pose and eye status from Face++
        headpose, eyestatus = await get_headpose_eyestatus_facepp(contents)

        # Get emotions from both APIs
        dominant_facepp, emotions_facepp = await get_emotion_facepp(contents)
        dominant_azure, emotions_azure = await get_emotion_azure(contents)

        # Prepare the response with labeled results
        response_data = {
            "Face++ Head Pose": headpose,
            "Face++ Eye Status": eyestatus,
            "Face++ Emotion": {
                "emotions": emotions_facepp,
                "dominantEmotion": dominant_facepp
            },
            "Azure Emotion": {
                "emotions": emotions_azure,
                "dominantEmotion": dominant_azure
            },
        }

        # Calculate consensus emotion if both APIs provided emotion data
        if emotions_facepp and emotions_azure:
            consensus_emotion, consensus_emotions = calculate_consensus(emotions_facepp, emotions_azure)
            response_data["Consensus Emotion"] = {
                "emotions": consensus_emotions,
                "dominantEmotion": consensus_emotion
            }

        # Generate the final human-readable output with labels
        output = f"You seem {response_data['Face++ Emotion']['dominantEmotion']}. How about something to lift your mood?\n"
        output += f"Face++ Head Pose: {response_data['Face++ Head Pose']}\n"
        output += f"Face++ Eye Status: {response_data['Face++ Eye Status']}\n"
        output += f"Face++ Emotion: {response_data['Face++ Emotion']['emotions']} (Dominant: {response_data['Face++ Emotion']['dominantEmotion']})\n"
        output += f"Azure Emotion: {response_data['Azure Emotion']['emotions']} (Dominant: {response_data['Azure Emotion']['dominantEmotion']})\n"
        
        if "Consensus Emotion" in response_data:
            output += f"Consensus Emotion: {response_data['Consensus Emotion']['emotions']} (Dominant: {response_data['Consensus Emotion']['dominantEmotion']})"

        return JSONResponse({"result": output})

    except requests.RequestException as e:
        logging.error(f"Request error: {str(e)}")
        return JSONResponse({"error": f"Request error: {str(e)}"}, status_code=400)
    except Exception as e:
        logging.error(f"Internal Server Error: {str(e)}")
        return JSONResponse({"error": f"Internal Server Error: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("faceazur:app", host="0.0.0.0", port=8002, reload=True)
