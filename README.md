RATOCER - Reichsanzeiger table of content extraction and recognition  
========================  
![Python 3.8](https://img.shields.io/badge/python-3.8-yellow.svg)
![license](https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg)


Overview
--------

**RATOCER** extracts and recongizes the table of content on Reichsanzeiger pages.

Building instructions
---------------------
Use the [Dockerfile](Dockerfile):

    $ docker build -t ratocer .
    $ docker run -it --rm -v "imgpath":/usr/src/app --user "$(id -u):$(id -g)"  ratocer main.py "imgpath"


Running
-------
Here is an example for extract the TOC withouth OCR:

    # perform deskewing, crop and splice of a page
    $ docker run -it --rm -v "imgpath":/usr/src/app --user "$(id -u):$(id -g)"  ratocer main.py --crop_only "imgpath"

Copyright and License
----

Copyright (c) 2021 Universitätsbibliothek Mannheim

Author: [Jan Kamlah](https://github.com/jkamlah)
