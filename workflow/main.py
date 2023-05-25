"""
main.py
====================================
The core mdafe
"""
import os
from pathlib import Path

def agg():
    """
    for each score file in score_path_list,
    identify best performing hyperparameter combination on validation set
    and collect corresponding metrics.

    save resulting table to new csv file stats_path
    """
    
def about_me(your_name):
    """
    Return the most important thing about a person.

    Args:
        your_name
            A string indicating the name of the person.
    """
    return "The wise {} loves Python.".format(your_name)


class ExampleClass:
    """An example docstring for a class definition."""

    def __init__(self, name):
        """
        Blah blah blah.
        Parameters
        ---------
        name
            A string to assign to the `name` instance attribute.
        """
        self.name = name

    def about_self(self):
        """
        Return information about an instance created from ExampleClass.
        """
        return "I am a very smart {} object.".format(self.name)








def aggregate(
    score_path_list: str,
    stats_path: str,
):
    """
    For each score file in score_path_list,
    identify best performing hyperparameter combination on validation set
    and collect corresponding metrics.

    save resulting table to new csv file stats_path

    Args:
        score_path_list: the path for the input score files
        stats_path: the path for the output stats csv file. which stores the best performing combinationbased on the the average coefficient of determination or accuracy on validation set

    """

    df_list = []

    # create empty token file for snakemake if empty (insufficient samples in dataset)
    if not df_list:
        Path(stats_path).touch()
        return



