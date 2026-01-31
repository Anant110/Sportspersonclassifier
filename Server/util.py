import joblib
import json
import numpy as np
import base64
import cv2
# In Server/util.py
from .wavelet import w2d


__class_name_to_number={}
__class_number_to_name={}

__model=None
def classify_image(image_base64_data,file_path=None):
    imgs=get_cropped_img_if_2_eyes(file_path,image_base64_data)

    result=[]
    for img in imgs:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        scalled_raw_img=cv2.resize(img,(32,32))
        img_har=w2d(img,'db1',5)

        scalled_img_har=cv2.resize(img_har,(32,32))
        combined_img=np.vstack((scalled_raw_img.reshape(32 * 32 * 3,1),scalled_img_har.reshape(32*32,1)))

        len_img_array=32*32*3+32*32

        final=combined_img.reshape(1,len_img_array).astype(float)

        # result.append(__class_number_to_name[__model.predict(final)[0]])
        result.append({
            'class':__class_number_to_name[__model.predict(final)[0]],
            'class_probability':np.round(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary':__class_name_to_number
        })

    return result

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json","r") as f:
        __class_name_to_number=json.load(f)
        __class_number_to_name={v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("./artifacts/celebrities_model.pkl","rb") as f:
            __model=joblib.load(f)
    print("Loading saved artifacts...done")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    if not b64str:
        return None
    if ',' not in b64str:
        return None
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_img_if_2_eyes(img_url,image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/harcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/harcascades/haarcascade_eye.xml')

    if img_url:
        face_img = cv2.imread(img_url)
    else:
        face_img=get_cv2_image_from_base64_string(image_base64_data)

    gray=cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    cropped_faces=[]
    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=face_img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_base64_image_for_dhoni():
    with open('base64.txt') as f:
        return f.read()


if __name__=='__main__':
    load_saved_artifacts()
    print(classify_image(get_base64_image_for_dhoni(),None))
    # print(class_number_to_name(0))
    print(classify_image(None,"./test_images/lionel3.jpg"))
    print(classify_image(None,"./test_images/fedrer1.jpg"))
    print(classify_image(None,"./test_images/serena1.jpg"))
    print(classify_image(None,"./test_images/virat1.jpg"))
