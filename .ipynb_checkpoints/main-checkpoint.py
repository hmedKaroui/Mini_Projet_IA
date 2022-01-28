import tensorflow as tf
from colabcode import ColabCode
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing.image import ImageDataGenerator
import uvicorn 
import numpy as np
from keras.preprocessing import image

class Image(BaseModel):
    imageName : str
        
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 128,
                                                 class_mode = 'categorical',
                                                 shuffle=True)
#model loading
cnn=tf.keras.models.load_model("./myModel_v1")

app = FastAPI()


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def read_root():
  return {"message":"welcome to first"}

@app.post('/predict')
def get_image_category(data: Image):
    received = data.dict()
    test_image = image.load_img('single_prediction/20 test for rapport/'+received['imageName'], target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    training_set.class_indices
    if result[0][0] == 0:
        prediction = 'CAN'
    else:
        prediction = 'NOR'
    return {'classificaton_result': prediction}