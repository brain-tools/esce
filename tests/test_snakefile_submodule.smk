from snakemake.utils import min_version


min_version("7.0")


configfile: workflow.source_path("../tests/test_config.yaml")
configfile: workflow.source_path("../config/style.yaml")
configfile: workflow.source_path("../config/grids.yaml")



# declare https://github.com/brain-tools/esce as a module
module esce:
    snakefile:
        github("brain-tools/esce", path="workflow/Snakefile", branch="dev")
    config:
        config


# use all rules from https://github.com/brain-tools/esce
use rule * from esce as esce_*

rule all:
    input:
        rules.esce_all.input,