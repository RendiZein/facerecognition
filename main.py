from fastapi import FastAPI, File, UploadFile, Depends
from pydantic import BaseModel
import cv2
import numpy as np
from face_recognition_onnx import *
import onnx
# import random
from sklearn.decomposition import PCA
import json
from json import JSONEncoder
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import model_d




if os.path.isfile("resnet100.onnx")==False: #download untuk pertama kali
    mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx')
    print("Download Model, Please wait")
model_name='resnet100.onnx'
print("Loading Model, Please wait")
model = get_model(model_name)

#Helper convert numpy to json
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
#Helper convert json to numpy
def json_numpy(encodedNumpyData):
    decodedArrays = json.loads(encodedNumpyData)
    finalNumpyArray = np.asarray(decodedArrays["array"])
    return finalNumpyArray



app = FastAPI()
model_d.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Defining path operation for root endpoint
@app.get("/")
async def root():
    return {"Coding Assignment"}

@app.get("/face")
def list_of_face(db: Session = Depends(get_db)):
    """
    Get a list of all faces (name) in the database.
    """
    nama_list=db.query(model_d.Face.id).all()
    nama_list=[i[0] for i in nama_list]
    if len(nama_list)==0:
        return {f"Database kosong"}
    else:
        return {f"Orang yang ada di databse adalah {nama_list}"}

@app.post("/face/register")
async def register(name:str,file: UploadFile,db: Session = Depends(get_db)):
    """
    Register new face and add the feature to database.
    The input must be image
    """
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    feature=detectandfeature(img, model)    
    if feature is None:
        return {f"filename {file.filename} gagal diinput"}
    else:
        numpyData = {"array": feature}
        feature_json = json.dumps(numpyData, cls=NumpyArrayEncoder)
        db_new=model_d.Face(id=name,feature = feature_json)
        # db_new2=model.Face(value=new_data.value)
        db.add(db_new)
        db.commit()
        db.refresh(db_new)
        # face_dict[name]=feature
        return {f"filename {file.filename} berhasil diinput"}
        

@app.post("/face/recognize")
async def recognize(file: UploadFile,db: Session = Depends(get_db)):
    """
    Recognize new face and by matcing the feature from database.
    The input must be image
    """
    #baca file
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #extract feature
    new_feature=detectandfeature(img, model)
    if new_feature is None: #handling tidak ada muka
        return
    #store feature sebagai list
    json_file=db.query(model_d.Face.feature).all()
    json_file=[i[0] for i in json_file]
    features=[json_numpy(i) for i in json_file]
    # features=[np.asarray(json.loads(i[0][0])['array']) for i in json_file]
    print(type(features[0]))
    #store name sebagai list
    names=db.query(model_d.Face.id).all()
    names=[i[0] for i in names]
    similiarity=np.array([])
    for feature in features:
        # Compute cosine similarity between embedddings
        similiarity = np.append(similiarity,np.dot(feature, new_feature.T))
    #lihat nilai terbesar
    most_matching_id=np.argmax(similiarity)
    #cek apakah melewati threshold
    if similiarity[most_matching_id]>0.2:
        return{f"Orang yang berada di gambar mirip dengan {names[most_matching_id]}"}
    else:
        return{"tidak ada yang mirip"}   
    
@app.delete("/face/{}")
def delete(del_name:str,db: Session = Depends(get_db)):
    """
    delete face from database.
    The input must be string (name of the face)
    """
    names=db.query(model_d.Face.id).all()
    names=[i[0] for i in names]
    global face_dict
    if del_name in names:
        del_item=db.query(model_d.Face).filter(model_d.Face.id == del_name).first()
        db.delete(del_item)
        db.commit()
        return {f"{del_name} berhasil dihapus"}
    else:
        return {f"Tidak ada {del_name} di database"}