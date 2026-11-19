"""Run with: streamlit run app.py -- --run-dir /path/to/trained/run"""
import argparse
from pathlib import Path
import io
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from road_surface.inference import load_trained_artifact, crop_region, predict_image

st.set_page_config(page_title='Road Surface Explorer',page_icon='❄️',layout='wide')
st.title('Road Surface Explorer')
st.write('Upload a road photo and select a region containing mostly pavement. The model was trained on close road-surface patches.')
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',default=str(Path(__file__).resolve().parent/'results/main/efficientnet_b0_seed42'))
args,_=parser.parse_known_args()
run_dir=args.run_dir

@st.cache_resource
def cached_artifact(path):return load_trained_artifact(path)

try:artifact=cached_artifact(run_dir)
except (ValueError,FileNotFoundError,OSError) as exc:
    st.info(str(exc));st.stop()
names={'custom_cnn':'Custom CNN','mobilenet_v2':'MobileNetV2','efficientnet_b0':'EfficientNetB0'}
st.sidebar.caption(f"Model: {names.get(artifact[1]['model'],artifact[1]['model'])}")
st.caption('Research prototype. Model scores are not calibrated probabilities. A photograph cannot confirm invisible black ice or measure tyre grip.')
st.expander('Class guide').write('Dry: visibly dry pavement. Wet: visibly damp or wet pavement. Snow: snow-covered surface. Ice: visually labelled icy surface. Slush: wet, melting snow mixed with water. Labels follow the Roads dataset; mixed or unclear surfaces may require manual review.')
upload=st.file_uploader('Upload a road photo',type=['jpg','jpeg','png'])
if upload:
    try:
        if upload.size>20*1024*1024:raise ValueError('Please upload an image smaller than 20 MB.')
        with Image.open(io.BytesIO(upload.getvalue())) as opened:
            if opened.width*opened.height>30_000_000:raise ValueError('Please use an image below 30 megapixels.')
            image=ImageOps.exif_transpose(opened).convert('RGB')
        left,right=st.slider('Horizontal region (% of image width)',0,100,(0,100))
        top,bottom=st.slider('Vertical region (% of image height)',0,100,(0,100))
        roi=(image.width*left//100,image.height*top//100,image.width*right//100,image.height*bottom//100)
        region=crop_region(image,roi)
        a,b=st.columns(2)
        a.image(image,caption='Original image',width='stretch');b.image(region,caption='Selected road region',width='stretch')
        if st.button('Classify selected region',type='primary'):
            result=predict_image(image,artifact,roi)
            st.subheader(result['class'].replace('_',' ').title())
            st.metric('Highest model score',f"{max(result['scores'].values()):.1%}")
            st.bar_chart(result['scores'])
            if result['low_score']:st.info('Low model score: this image needs manual review.')
            st.caption('Mixed surfaces, unusual cameras, shadows and compression may change the prediction.')
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError) as exc:st.error(str(exc))
