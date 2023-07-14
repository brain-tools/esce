.. Empirical Sample Complexity Estimator documentation master file, created by
   sphinx-quickstart on Wed May 01 02:39:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Empirical Sample Complexity Estimator's documentation!
==================================================================

.. Release v\ |version|. (:ref:`Installation <install>`)

**The Empirical Sample Complexity Estimator** provides a Snakemake workflow to analyze the performance of machine learning models as the sample size increases. The goal of the workflow is to make it easy to compare the scaling behaviour of various machine learning models, feature sets, and target variables.

For more information, refer to the following publication: `Schulz, M. A., Bzdok, D., Haufe, S., Haynes, J. D., & Ritter, K. (2022). Performance reserves in brain-imaging-based phenotype prediction. BioRxiv, 2022-02. <https://biorxiv.org/content/10.1101/2022.02.23.481601v1.full>`_.


Features
-----------------------------
.. .. automodule:: workflow.main
..     :members:

.. automodule:: esce
    :members:

.. .. automodule:: workflow.scripts.aggregate
..     :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Example workflow
-----------------
To try the example workflow, check out the example/ directory and run the following command:

>>> snakemake --cores 1 --configfile example/example_config.yaml --rerun-triggers mtime --use-conda --rerun-incomplete all

Once the workflow has completed, you can view the results in the results/example-dataset/statistics directory and the plots in results/example-dataset/plots.


The User Guide
--------------

This part of the documentation, which is mostly prose, begins with some background information, then focuses on step-by-step instructions.

.. .. toctree::
..    :maxdepth: 2

..    user/install


Configuration
--------------
For more information on the configuration file, see:
.. .. toctree::
..    :maxdepth: 2

..    config

Features
-----------------------------

If you are looking for information on a specific function, class, or method,
this part of the documentation is for you.

.. .. toctree::
..    :maxdepth: 2

..    feature


The Contributor Guide
---------------------

If you want to contribute to the project, this part of the documentation is for
you.

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

.. .. toctree::
..    :maxdepth: 3

..    dev/contributing


Support
-------

If you are having issues, please let us know. @email


License
-------

The project is licensed under the ... license.

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`