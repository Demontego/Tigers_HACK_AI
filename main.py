import streamlit as st
import pandas as pd
import cv2
import tempfile
import numpy as np
import onnxruntime as ort
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.config import get_cfg
from torchvision import  transforms
import torch
import os
from model import img_CNN, img_siamese

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.MODEL.WEIGHTS = "model_final.pth"
my_metadata = Metadata(name= "cats_train")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
my_metadata.set(thing_classes = ['Tiger', 'Leopard'])# Let training initialize from model zoo
predictor = DefaultPredictor(cfg)

data_transforms ={'c': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    's': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}            

ort_session = ort.InferenceSession("best_model.onnx")
labels = ['Дальневосточный леопард', "Тигр", 'Никто']

siamese_model = torch.load('siamese.pth', map_location=device)
etalon = pd.read_csv('princess_data.csv').sample(10)
etal_image = []
for i in etalon.id.values:
    img = Image.open('siamese/Princess/'+i).convert('RGB')
    img = data_transforms['s'](img)
    etal_image.append(img.unsqueeze(0))
etal_image = torch.cat(etal_image, 0).to(device)


st.set_page_config(
   page_title="Ex-stream-ly Cool App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)
max_width_str = f"max-width: 1920px;"
st.markdown(
	f"""
		<style>
			.reportview-container .main .block-container {{{max_width_str}}}
		</style>    
	""",
	unsafe_allow_html=True
)


def classification_photo(filesImage):
    my_bar = st.progress(0)
    tmp = 100//len(filesImage)
    for percent_complete, file in enumerate(filesImage):
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = Image.fromarray(img[:,:,::-1]).convert("RGB")
        img = data_transforms['c'](img).unsqueeze(0).numpy()
        outputs = ort_session.run(
            None,
            {'input.1': img},
        )
        cl = labels[np.argmax(outputs[0])]
        st.markdown(cl)
        my_bar.progress(int((percent_complete+1) * tmp))

def detect_princess(fileImage):
    file_bytes = np.asarray(bytearray(fileImage.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    outputs = predictor(img)
    preds=outputs["instances"].to("cpu")
    classes = preds.pred_classes.numpy()
    boxes = preds.pred_boxes.tensor.detach().cpu().numpy().astype(int)
    for i in range(len(classes)):
        if classes[i]==0:
            img = img[boxes[i][1]:boxes[i][3],boxes[i][0]:boxes[i][2],::-1]
            img = data_transforms['s'](Image.fromarray(img))
            test_image = torch.cat([img.unsqueeze(0) for _ in range(10)], 0).to(device)
            pred = siamese_model(etal_image,test_image)
            result = np.mean(torch.argmax(pred, dim=1).detach().cpu().numpy())
            st.markdown("С вероятностью "+str(int(result*100))+"% Это принцесса")
        else:
            st.markdown("принцессы нет")
        return
    st.markdown("Животных нет")
            
    
    

def detect_on_photo(filesImage):
    my_bar = st.progress(0)
    tmp = 100//len(filesImage)
    for percent_complete, file in enumerate(filesImage):
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1],
                   metadata=my_metadata, 
                   scale=1,
                   instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(v.get_image())
        my_bar.progress(int((percent_complete+1) * tmp))
        
def detect_on_video(fileVideo):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(fileVideo.read())
    video = cv2.VideoCapture(tfile.name)
    # stframe = st.empty()
    #polling
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    result = cv2.VideoWriter('output.mp4',
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps=float(frames_per_second), frameSize=(width, height) ,isColor=True)
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break
        outputs = predictor(frame)
        v = Visualizer(frame[:, :, ::-1],
                   metadata=my_metadata, 
                   scale=1,
                   instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result.write(v.get_image())
    video.release()
    result.release()
    with open("output.mp4", "rb") as file:
         btn = st.download_button(
                 label="Download video",
                 data=file,
                 file_name="output.mp4",
                 mime="video/mp4"
               )
    if btn:
        st.write('Thanks for downloading!')
        # my_bar.progress(int((percent_complete+1) * tmp))

st.title("Демонстрация распознавания больших кошечек")
filesImage = st.file_uploader(label = "Детекция на фото", type=['png','jpg'], accept_multiple_files=True)
classImage = st.file_uploader(label = "Классификация фото", type=['png','jpg'], accept_multiple_files=True)
fileVideo = st.file_uploader(label = "Детекция на видео", type='mp4', accept_multiple_files=False)
Princess = st.file_uploader(label = "Определение принцессы", type=['png','jpg'], accept_multiple_files=False)


if filesImage:
    detect_on_photo(filesImage)
if classImage:
    classification_photo(classImage)
if fileVideo:
    detect_on_video(fileVideo)
if Princess:
    detect_princess(Princess)
# with st.spinner('Wait for it...'):
#     time.sleep(5)
# st.success('Done!')
