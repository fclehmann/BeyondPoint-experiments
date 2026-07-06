import argparse
from pathlib import Path
from cifar10_clf.dataset import LazyCIFAR10Combined

def main():
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 data.")
    parser.add_argument(
        "data_path", type=str, help="Path to the raw CIFAR-10 data directory"
    )
    
    # Parse arguments
    args = parser.parse_args()

    path_data = Path(args.data_path)  # Use the provided data path
    print(f"Data path: {path_data}")
    
    path_root_raw = path_data / "raw"
    path_root_processed = path_data / "processed" 

    path_root_raw.mkdir(exist_ok=True, parents=True)
    path_root_processed.mkdir(exist_ok=True, parents=True)
    
    # Call the function to process the CIFAR-10 dataset
    LazyCIFAR10Combined.from_original_cifar_10(
        path_root_raw=path_root_raw,
        path_root_processed=path_root_processed
    )

if __name__ == "__main__":
    main()