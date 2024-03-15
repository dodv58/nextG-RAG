# nextG-RAG

## Requirements

- Python version >= 3.9
- Download a Llama2 model and places in models/ (could be downloaded from https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF) 
- change LLM_MODEL variable in app.py to the model name.

### Using virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies 
`pip install -r requirements.txt`

## Run app
`python app.py`

## Routes 

### Upload files
`127.0.0.1:5000/files`

### Question answering
`127.0.0.1:5000/qna`



# commands to install

```
$ sudo apt install build-essential
$ git clone https://github.com/dodv58/nextG-RAG.git
$ cd nextG-RAG 
$ mkdir models
$ cd models
$ wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q6_K.gguf
$ cd ..
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

# Run this command to use GPU 
$ CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

$ python app.py
```