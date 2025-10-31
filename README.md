# Introduction
This is simple "local RAG" that allows searches to local files which have been tokenized to local 
vector database, and web content that has likewise been tokenized to local vector database.

# Installation and preparation
First install Tesseract with language support, in MacOS using brew for example:
```
brew install tesseract
brew install tesseract-lang
```

You might also want to install HuggingFace CLI in order to make working with models easier:
```
brew install huggingface-cli
```

You probably also need to set the following to avoid some errors and warnings:
```
export TESSDATA_PREFIX=/opt/homebrew/share/tessdata
export TOKENIZERS_PARALLELISM=true
```

You need to install requirements first. I suggest a clean virtual environment for this.
```
pip install -r requirements.txt
playwright install chromium
```

You need to have a suitable LLM model in GGUF format downloaded. Its path needs to speacified
in chat_ui.py file. You need to strike a balance between speed and quality when selecting the model,
given your computer's performance. 7B Q4 quantified models  provide decent answers, and produce output relatively 
with suitably powerful laptop. For example:
- mistral-7b-instruct-v0.2.Q4_K_M.gguf (good mainly for English material)
- llama-2-13b-chat.Q4_K_M.gguf (manages with several languages)

You also need to download transformer models to local directory first to run totally offline. The following two
seem to perform decently, with multilingual (somewhat unsurprisingly) being better with non-English material.
```
python download_transformers.py
hf download intfloat/multilingual-e5-base --local-dir ~/models/e5-base
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ~/models/minilm
```

# Tokenizing PDF material
The first step is tokenizing the material for you searches. This happens as follows:
1. Download, print to PDF or whatever, and place PDF material to suitable directory structure
2. Add these directories to ingest.py DIRS_TO_PROCESS array
3. Run ingest.py

ingest.py requires one or more directory names. The directory names should follow theme_language format.
Language needs to be specified with 2-letter ISO code ("en", "fi" etc.)
For example, if you have car related documents in English, you could invoke as follows:
```
python ingest.py /Users/myusername/Documents/Cars --lang en --collection Cars
```

This should create a local Chroma DB with embeddings of ingested PDF files, page by page.
Embedding happens using the language designated in directory name.
Running ingestion will take some time, depending on the number and size of the files and
the performance of your computer. As an example, processing 28 PDF files, each of which is 200 to 400
pages long, takes in the order of 10 minutes in powerful M4 MacBook Pro, and produces 150 MB Chroma DB.

If there is limit to limit results to certain audiences, you can specify "audiences". For example, if you want
to control access to documents regarding your network, you can specify audiences like "network" and "ops" and "admin",
implying that these documents are restricted to these audiences. By default audience
is "public", meaning that everybody can access ingested documents. UI searches by default for public results,
but you can select an audience to extend the public search with results limited to the given audience(s).
```
python ingest.py /Users/myusername/Documents/Network_documentation --lang en --Collection "Network_documentation" --audiences network ops admin
```

# Tokenizing web material
Web material is ingested by crawling through web site in some language recursively, transforming web pages
to text and then tokenizing the text. For example, to ingest web pages of Kela, do the following:
```
python ingest.py https://www.kela.fi --collection Kela --lang fi
```

In some cases interesting material is not linked from top page. For these cases we can try "ranged fetch" that
dynamically builds URLs and fetches contents. Use parameters "range" and "width" here:
```
python ingest.py "https://www.terveyskirjasto.fi/dlk" --collection Terveys --lang fi --range 1 1425 --width 5
```

This will crawl web addresses 
```
https://www.terveyskirjasto.fi/dlk00001
...
https://www.terveyskirjasto.fi/dlk01425
```

Some sites require authentication. For a hopefully general solution, we support reading in cookies file and
crawling a site through "cookie session". The cookies you can catch from your browser after logging in. An example shortly.
You can also specify a "cookie string" in this format
```
--cookies "sessionid=abc123; csrftoken=xyz987"
```

As with PDF ingestion, you can specify one or more "audiences" for ingested pages. If you don't specify an audience,
the audience will be "public". For example, reading in Nitor intra (requires cookies)
```
python ingest.py "https://nitor.atlassian.net/wiki/home" --collection NitorIntra --lang en --audiences nitorean -cookies ../cookies.txt
```

There are surely other types of "difficult to crawl" sites. You may have to figure out their logic yourself and
may be forced to augment the fetching logic provided. Naturally, you need to respect copyrights accordingly.
For example, it is not actually completely OK to read the whole Terveyskirjasto above since terms of use prohibit making 
copies of the information. They do allow educational use give the source is given (which we do here since we link
to the original). Many sources also require authorization, which we do not currently do.

The time it takes for ingestion to complete is dependent on many things, including speed of your Internet connection,
the speed of web page host's Internet connection, the number of web pages found in the target site
and the power of your computer. For large sites this can take plenty of time and create a large database.

# Running queries on material
Chat-like graphical interface allows user to make searches to tokenized material, and follow the
provided references to source material (pages of PDF files, or links to web pages). For PDF files 
we need access to source material as well. The source to use is decided based on the "theme" that you choose
(which actually selects your Chroma database) and the language of your question (which is automatically detected).
Furthermore, you can a specify an "audience" which defaults to "public". Results are filtered so that if 
audience is given, you will also see results permitted to given audience. Everybody will see results from
"public" sources, that is, material that is not restricted to any audience.

Running
```
python ui.py
```

After this, you can move to [http://localhost:7860](http://localhost:7860/).

You will get the LLM answer on lower left.

You will get a list of references on right hand side, with the first
reference selected (the page of the PDF document referenced,). You can choose difference reference, and 
move within the selected PDF file.
