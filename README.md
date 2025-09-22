# Introduction
This is simple "local RAG" that allows searches to local files which have been tokenized to local 
vector database.

# Installation and preparation
You need to install requirements first. I suggest a clean virtual environment for this.
```
pip install -r requirements.txt
```

You need to have a suitable LLM model in GGUF format downloaded. You need to strike a 
balance between speed and quality, given your computer's performance. 7B Q4 quantified models
at provide decent answers, and produce output relatively with suitably powerful laptop. For example:
- mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Tokenizing PDF material
The first step is tokenizing the material for you search. This happens as follows:
1. Download, print to PDF or whatever, and place PDF material to suitable directory structure
2. Add these directories to ingest.py DIRS_TO_PROCESS array
3. Run ingest.py

This should create a local Chroma DB with embeddings of ingested PDF files, page by page. 
Running ingestion will take some time, depending on the number and size of the files and
the performance of your computer. As an example, processing 28 PDF files, each of which is 200 to 400
pages long, takes in the order of 10 minutes in powerful M4 MacBook Pro, and produces 150 MB Chroma DB.

# Running queries on material
There are two ways to make queries on material:
1. Simple text based interface
2. "Chat GPT" style UI with possibility to navigate to source PDFs

Both of these require that you specify where your model file is located. You need to 
change LLAMA_MODEL_PATH in query.py / chat_ui.py.

## Text based interface
```
python query.py
```

Basically, you present a question and will get an answer referencing the material using page numbers.
Some performance data is also displayed.

```
Query> Is service-oriented architecture dead?
llama_perf_context_print:        load time =    1003.72 ms
llama_perf_context_print: prompt eval time =    1003.15 ms /   786 tokens (    1.28 ms per token,   783.53 tokens per second)
llama_perf_context_print:        eval time =    1985.59 ms /   138 runs   (   14.39 ms per token,    69.50 tokens per second)
llama_perf_context_print:       total time =    3004.87 ms /   924 tokens
llama_perf_context_print:    graphs reused =        133

--- Answer ---
Based on the provided passages, service-oriented architecture (SOA) is a design approach that emerged 
in the late 1990s to promote reusability and collaboration among multiple services through network 
communication. SOA is different from microservices, which gained popularity in the mid-2010s to 
address modern systems' need to change quickly, scale, and fit distributed cloud computing naturally. 
The passages do not indicate that SOA is dead; instead, they discuss its origins and differences from 
microservices. (fundamentalsofsoftwarearchitecture.pdf p183, buildingmicroservices2ndedition.pdf p30)

Query>
```

## Chat-like interface
Chat-like graphical interface allows user to make searches to tokenized material, and follow the
provided references to source material (pages of PDF files). So we need access to source material
as well - you need to list the PDF directories id DOC_DIRS of chat_ui.py.

Running
```
python chat_ui.py
```

After this, you can move to [http://localhost:7860](http://localhost:7860/).

There is a sample question in "Question" field. You can try with that, or write your own question.

You will get the LLM answer on lower left, and list of references on right hand side, with the first
references selected (the page of PDF document referenced). You can choose difference reference, and 
move within the selected PDF file.