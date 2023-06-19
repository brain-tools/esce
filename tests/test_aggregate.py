from workflow.scripts.aggregate import aggregate
import os
import pandas as pd


def test_aggregate():
    # Create files for testing
    num_of_files = 5
    score_files = ['tests/data/aggregate_input_' + str(i) + '.csv' for i in range(1,num_of_files + 1)]
    stats_file = 'tests/data/aggregate_output.csv'

    # # # 
    # Test case 1: Generate sample data for score files: for r2_val
    # # #
    sample_data = [
        {
            "n": 100,
            "s": 1,
            "r2_val": 0.0,
            "acc_val": 0.9
        },
        {
            "n": 100,
            "s": 2,
            "r2_val": 0.1,
            "acc_val": 0.91
        },
        {
            "n": 200,
            "s": 1,
            "r2_val": 0.2,
            "acc_val": None
        },
        {
            "n": 200,
            "s": 1,
            "r2_val": 0.3,
            "acc_val": None
        },
        {
            "n": 200,
            "s": 3,
            "r2_val": 0.5,
            "acc_val": None
        },
        {
            "n": 100,
            "s": 1,
            "r2_val": 0.6,
            "acc_val": None
        },
        {
            "n": 100,
            "s": 1,
            "r2_val": 0.7,
            "acc_val": 0.1
        },
        {
            "n": 200,
            "s": 2,
            "r2_val": 0.4,
            "acc_val": None
        },
        {
            "n": 200,
            "s": 2,
            "r2_val": 0.3,
            "acc_val": 0.9
        },
    ]

    # Write sample data to score files
    for i, file in enumerate(score_files):
        pd.DataFrame(sample_data[i::num_of_files]).to_csv(file, index=False)

    # Run the function being tested
    aggregate(score_files, stats_file)

    # Read the resulting stats file
    result_df = pd.read_csv(stats_file)

    # Define the expected output based on the sample data
    expected_data = [
        {
            "n": 100,
            "s": 1,
            "r2_val": 0.7,
            "acc_val": 0.1
        },
        {
            "n": 100,
            "s": 2,
            "r2_val": 0.1,
            "acc_val": 0.91
        },
        {
            "n": 200,
            "s": 1,
            "r2_val": 0.3,
            "acc_val": None
        },
        {
            "n": 200,
            "s": 2,
            "r2_val": 0.4,
            "acc_val": None
        },
        {
            "n": 200,
            "s": 3,
            "r2_val": 0.5,
            "acc_val": None
        }
    ]
    expected_df = pd.DataFrame(expected_data)

    assert result_df.equals(expected_df), "Expected best n, s aggregated combination calculated incorrectly for r2_val"

    # Remove temporary files
    for file in score_files + [stats_file]:
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)

    # # # 
    # Test case 2: Generate sample data for score files: for r2_val
    # # #
    sample_data_acc = [
        {
            "n": 100,
            "s": 1,
            "acc_val": 0.9
        },
        {
            "n": 100,
            "s": 2,
            "acc_val": 0.91
        },
        {
            "n": 200,
            "s": 1,
            "acc_val": 0.2
        },
        {
            "n": 200,
            "s": 1,
            "acc_val": 0.3
        },
        {
            "n": 200,
            "s": 3,
            "acc_val": 0.5
        },
        {
            "n": 100,
            "s": 1,
            "acc_val": 0.6
        },
        {
            "n": 100,
            "s": 1,
            "acc_val": 0.1
        },
        {
            "n": 200,
            "s": 2,
            "acc_val": 0.4
        },
        {
            "n": 200,
            "s": 2,
            "acc_val": 0.9
        },
    ]

    for i, file in enumerate(score_files):
        pd.DataFrame(sample_data_acc[i::num_of_files]).to_csv(file, index=False)

    aggregate(score_files, stats_file)

    result_df = pd.read_csv(stats_file)

    expected_data = [
        {
            "n": 100,
            "s": 1,
            "acc_val": 0.9
        },
        {
            "n": 100,
            "s": 2,
            "acc_val": 0.91
        },
        {
            "n": 200,
            "s": 1,
            "acc_val": 0.3
        },
        {
            "n": 200,
            "s": 2,
            "acc_val": 0.9
        },
        {
            "n": 200,
            "s": 3,
            "acc_val": 0.5
        }
    ]
    expected_df = pd.DataFrame(expected_data)

    assert result_df.equals(expected_df), "Expected best n, s aggregated combination calculated incorrectly for acc_val"

    # Clean up temporary files
    for file in score_files + [stats_file]:
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)


    # # # 
    # Test case 3: Check empty file is always created to run Snakemake workflow smoothly
    # # # 
    sample_data_empty = [
        {
            "n": None,
            "s": None,
            "r2_val": None,
            "acc_val": None,
        } 
        for _ in range(num_of_files)
    ]

    for i, file in enumerate(score_files):
        pd.DataFrame(sample_data_empty[i::num_of_files]).to_csv(file, index=False)

    aggregate(score_files, stats_file)

    assert os.path.exists(stats_file), "Empty file not created for Snakemake workflow when empty csvs were inputted"

    for file in score_files + [stats_file]:
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)