"""
python3 convert_to_mds_parallel.py \
--input_dir=/gcs/mlperf_llama405b/preprocessed/ \
--mds_output_dir=/gcs/mlperf_llama405b/mds_dataset \
--max_examples=100
"""

import os
import argparse
import ray
import numpy as np
from megatron.core.datasets.indexed_dataset import IndexedDataset
from streaming import MDSWriter
@ray.remote
def convert_to_mds(dataset_prefix, mds_output_dir, max_examples=None):
    """
    Converts a single dataset to MDS format.

    Args:
        dataset_prefix (str): The path prefix for the input dataset.
        mds_output_dir (str): The directory to write the MDS dataset to.
        max_examples (int, optional): The maximum number of examples to process. Defaults to None.
    """
    try:
        dataset = IndexedDataset(path_prefix=dataset_prefix)
        print(f"Successfully loaded dataset with prefix: {dataset_prefix}")

        columns = {'tokens': 'ndarray:uint16'}

        # Extract the base name from the dataset_prefix to create a unique subdirectory for each conversion
        base_name = os.path.basename(dataset_prefix)
        output_path = os.path.join(mds_output_dir, base_name)

        with MDSWriter(out=output_path, columns=columns) as writer:
            for i, sample in enumerate(dataset):
                if max_examples is not None and i >= max_examples:
                    break
                sample_dict = {
                    "tokens": sample
                }
                writer.write(sample_dict)
        return f"Successfully converted {dataset_prefix} to MDS format at {output_path}"
    except FileNotFoundError:
        return f"Error: Could not find the dataset files for prefix: {dataset_prefix}"
    except Exception as e:
        return f"An unexpected error occurred while processing {dataset_prefix}: {e}"

def list_dataset_prefixes(input_dir):
    """
    Lists all dataset prefixes in a given directory.

    Args:
        input_dir (str): The directory to search for dataset files.

    Returns:
        list: A list of unique dataset prefixes.
    """
    prefixes = set()
    for filename in os.listdir(input_dir):
        if filename.endswith(".bin"):
            prefix = os.path.splitext(filename)[0]
            prefixes.add(os.path.join(input_dir, prefix))
    return list(prefixes)

def main():
    parser = argparse.ArgumentParser(description="Convert multiple datasets to MDS format in parallel using Ray.")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="The directory containing the dataset files to be converted."
    )
    parser.add_argument(
        '--mds_output_dir',
        type=str,
        required=True,
        help="The directory where the MDS datasets will be stored."
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=None,
        help="The maximum number of examples to process for each file."
    )
    args = parser.parse_args()

    ray.init()

    dataset_prefixes = list_dataset_prefixes(args.input_dir)
    if not dataset_prefixes:
        print(f"No dataset files (.bin) found in {args.input_dir}")
        return

    futures = [convert_to_mds.remote(prefix, args.mds_output_dir, args.max_examples) for prefix in dataset_prefixes]
    results = ray.get(futures)

    for result in results:
        print(result)

if __name__ == '__main__':
    main()
