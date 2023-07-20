from snakemake.utils import min_version


min_version("7.0")


configfile: workflow.source_path("../tests/test_config.yaml")
configfile: workflow.source_path("../config/style.yaml")
configfile: workflow.source_path("../config/grids.yaml")



# declare https://github.com/brain-tools/esce as a module
module esce:
    snakefile:
        github("brain-tools/esce", path="workflow/Snakefile", branch="revert_esce_modularisation")
    config:
        config


# use all rules from https://github.com/brain-tools/esce
use rule * from esce as esce_*


rule prepare_data:
    input:
        [
            config["custom_datasets"][dataset][category][feature]
            for dataset in config["custom_datasets"]
            for category in config["custom_datasets"][dataset]
            for feature in config["custom_datasets"][dataset][category]
        ],


rule run:
    input:
        rules.esce_all.input,