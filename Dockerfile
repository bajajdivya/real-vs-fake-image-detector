FROM tensorflow/tensorflow
# RUN mkdir /inception
WORKDIR /app
ADD real_vs_fake_faces_project/requirements.txt /app/requirements.txt
CMD pip install -r requirements.txt 
ADD  inception.py /app/inception.py
# RUN python inception.py