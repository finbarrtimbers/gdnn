Dependencies/requirements:

- A sane linux, unix, or Mac OS X environment with python2.7 and a C++ compiler
- gnumpy and one of cudamat or npmat
- numpy
- nose (for tests)

Build:

Just type 
$ make
at your prompt to build the C++ components.


Running tests:
nosetests test_gdnn.py

Only a couple tests right now.


Running examples:

$ python mnistExample.py
$ python skipGramLogLinExample.py

Both examples will try to download the data they need. If something
goes wrong in that process you can do it yourself and symlink to the
files they check for.
