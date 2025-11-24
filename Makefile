SHELL := /bin/bash

all: prep-venv clean

prep-venv: 
	python3.11 -m venv fase7_env && \
	source fase7_env/bin/activate && \
	python3.11 -m pip install --upgrade pip && \
	python3.11 -m pip install -r requirements.txt  \
	&& /bin/bash

clean:
	-/bin/rm -rf fase7_env
