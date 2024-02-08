from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

import easyocr
import numpy as np
import tensorflow as tf
graph = tf.get_default_graph()

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model


async def object_detection(image, model):
    # Read image
    image1 = image.copy()
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = image1.resize((224,224))

    # Data preprocessing
    image_arr_224 = img_to_array(image1)/255.0 # Convert to array & normalized
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    
    # Make predictions
    coords = model.predict(test_arr)

    # Denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    return image,  coords

def ocr_it(image, region_threshold, img, cods):
    # Full image dimensions
    height = image.shape[0]
    
    #ROI
    img = np.array(img)
    xmin ,xmax,ymin,ymax = cods[0]
    roi = img[ymin:ymax,xmin:xmax]
    
    #Apply OCR to ROI
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(roi)

    # Apply ROI filtering and OCR
    text = [] 
    rectangle_size = roi.shape[0]*roi.shape[1]
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length*height / rectangle_size > region_threshold:
            text.append(result[1])
    # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    # plt.show()
    print(text)
    return text, roi

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    anpr_sys = load_model('./models/Object_Detection.h5')
    ml_models["anpr_sys"] = anpr_sys
    print('Model loaded Sucessfully')
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

def generatePIL(base64_string):
    try:
        decoded_bytes = base64.b64decode(base64_string)
        image_bytes = BytesIO(decoded_bytes)
        image = Image.open(image_bytes)
        return image
    except Exception as e:
        return None

class Footage(BaseModel):
    base64: str

app = FastAPI(lifespan=lifespan)


templates = Jinja2Templates(directory="templates")

@app.post("/push")
async def push_image(footage: Footage):
    img = generatePIL(footage.base64)
    if(img):
        image, cods = await object_detection(img,  ml_models["anpr_sys"])
        region_threshold=0.05
        text, region = ocr_it(image, region_threshold, img,cods)
        if len(text) == 0:
            return {"output": None}
        return {"output":text[0]}
    else:
        return {"output": None}

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={'request':request})


