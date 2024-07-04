from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from uuid import UUID
import cv2
import sys
import numpy as np
import mxnet as mx
import os
from scipy import misc
import onnx
import random
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans
import matplotlib.pyplot as plt
from mxnet.contrib.onnx.onnx2mx.import_model import import_model

###soon tobe deleted
def get_model(model):
    image_size = (112,112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    # model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model = mx.mod.Module(symbol=sym, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model
def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    # Assert input shape
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2
        assert image_size[0]==112
        assert image_size[0]==112 or image_size[1]==96
    
    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped
    
    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    
def get_input(detector,face_img):
    # Pass input images through face detector
    ret = detector.detect_face(face_img, det_type = 0)
    if ret is None:
        return None
    bbox, points = ret
    if bbox.shape[0]==0:
        return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    # Call preprocess() to generate aligned images
    nimg = preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

def face_detection(img):
    det_threshold = [0.6,0.7,0.8]
    mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')
    # detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)
    detector = MtcnnDetector(model_folder=mtcnn_path, num_worker=1, accurate_landmark = True, threshold=det_threshold)
    pre = get_input(detector,img)
    return pre

def get_feature(model,aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

def detectandfeature(img, model):
    # face_box=face_detection(img)
    # feature=get_feature(model,face_box)
    # return feature
    try:
        face_box=face_detection(img)
        feature=get_feature(model,face_box)
        return feature
    except :
        print("Tidak ada muka")
        return

###inti
face_dict=dict()
model_name='resnet100.onnx'
print("Loading Model, Please wait")
model = get_model(model_name)


###
indeks=[2,3,4]
value=[2**2,3**2,4**2]

app = FastAPI()



# Defining path operation for root endpoint
@app.get("/face")
def list_of_face():
    """
    Get a list of all faces (name) in the database.
    """
    return {f"The face is {list(face_dict.keys())}"}

@app.post("/face/register")
async def register(name:str,file: UploadFile):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    feature=detectandfeature(img, model)
    if feature is None:
        return {f"filename {file.filename} gagal diinput"}
    else:
        face_dict[name]=feature
        return {f"filename {file.filename} berhasil diinput"}
        

@app.post("/face/recognize")
async def recognize(file: UploadFile):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    new_feature=detectandfeature(img, model)
    if new_feature is None:
        return
    features=list(face_dict.values())
    names=list(face_dict.keys())
    similiarity=np.array([])
    for feature in features:
        # Compute cosine similarity between embedddings
        similiarity = np.append(similiarity,np.dot(feature, new_feature.T))
    #lihat nilai terbesar
    most_matching_id=np.argmax(similiarity)
    if similiarity[most_matching_id]>0.2:
        return{f"Orang yang berada di gambar mirip dengan {names[most_matching_id]}"}
    else:
        return{"tidak ada yang mirip"}   
    
@app.delete("/face/{}")
def delete(name:str):
    global face_dict
    if name in list(face_dict.keys()):
        face_dict=face_dict.pop(name)
        return {f"{name} berhasil dihapus"}
    else:
        return {f"tidak ada {name} di data"}