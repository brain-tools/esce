.. _install:

Development installation
--------------------------------
These are development setup and standards that are followed to by the core development team. If you are a contributor, it might a be a good idea to follow these guidelines as well.


Requirements
--------------------------------
A development setup can be hosted by your laptop, in a VM, on a separate server etc. Any such scenario should work fine, as long as it can satisfy the following:

* Is Unix-like system (Linux, BSD, Mac OSX) which supports Docker. Windows systems should have WSL+Docker or Docker Desktop.

* Has 10 GB or more of free disk space on the drive where Docker’s cache and volumes are stored. If you want to experiment with customizing Docker containers, you’ll likely need more.

* Can spare 2 GB of system memory for running Read the Docs, this typically means that a development laptop should have 8 GB or more of memory in total.

* Your system should ideally match the production system which uses the latest official+stable Docker distribution for Ubuntu (the docker-ce package). If you are on Windows or Mac, you may also want to try Docker Desktop.


Set up your environment
--------------------------------
1. Clone the repository:

>>> git clone https://github.com/brain-tools/esce.git

Once you have a copy of the project:
>>> cd esce

2. Installing Snakemake

In order to use this workflow, you must have either Anaconda or Miniconda installed and Snakemake must be installed. To install Snakemake, run the following command:

>>> conda install -c bioconda snakemake


.. To install, simply run this simple command in your terminal of choice::

..     $ python -m pip install requests

Get the Source Code
-------------------

Requests is actively developed on GitHub, where the code is
`always available <https://github.com/psf/requests>`_.

You can either clone the public repository::

    $ git clone https://github.com/psf/requests.git

Or, download the `tarball <https://github.com/psf/requests/tarball/main>`_::

    $ curl -OL https://github.com/psf/requests/tarball/main
    # optionally, zipball is also available (for Windows users).

Once you have a copy of the source, you can embed it in your own Python
package, or install it into your site-packages easily::

    $ cd requests
    $ python -m pip install .