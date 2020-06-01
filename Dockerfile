FROM bamos/openface

RUN ln -s /root/openface/models/ /models

RUN apt-get update && apt install python-pip

RUN pip install -U pip && pip install tqdm
