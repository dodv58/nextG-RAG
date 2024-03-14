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
