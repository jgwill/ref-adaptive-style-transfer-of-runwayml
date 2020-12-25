FROM guillaumeai/tf:1.12.0-cpu

COPY . /model
#RUN git clone https://github.com/jgwill/ref-adaptive-style-transfer-of-runwayml.git /model

WORKDIR /model

