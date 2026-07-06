import argparse
import pathlib

import simpsons.dataset
import torchvision
import PIL.Image

def main():
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Process simpsons data.")
    parser.add_argument(
        "data_path", type=str, help="Path to the raw simpsons data directory"
    )
    
    # Parse arguments
    args = parser.parse_args()

    path_data = pathlib.Path(args.data_path)  # Use the provided data path
    print(f"Data path: {path_data}")
    
    path_data_raw = path_data / "raw"
    path_data_processed = path_data / "processed" / "simpsons_64x64"


    dataset = simpsons.dataset.SimpsonsDataset(
        root=path_data_raw / "simpsons_dataset",
        transform=torchvision.transforms.Resize(
            size=(64, 64),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
    )

    path_data_processed.mkdir(parents=True, exist_ok=True)

    class_idx_to_name = dict(zip(dataset.class_to_idx.values(), dataset.class_to_idx.keys()))

    for idx, (raw_sample_path, class_idx) in enumerate(dataset.samples):
        class_label = class_idx_to_name[class_idx]
        img: PIL.Image.Image = dataset[idx][0]
        output_path = path_data_processed / class_label
        output_path.mkdir(exist_ok=True)
        img.save(path_data_processed / class_label / pathlib.Path(raw_sample_path).name)
        if idx % 1000 == 0:
            print(f"Progress: processed {idx}")

if __name__== "__main__":
    main()