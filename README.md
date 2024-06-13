# HR-Extreme Dataset

## Overview
HR-Extreme is a dataset containing high-resolution feature maps of physical variables for evaluating the performance of cutting-edge models on extreme weather prediction. This dataset focuses on 17 types of extreme weather events spanning the year 2020, based on HRRR data. The dataset is designed to support researchers in weather forecasting, ranging from physical methods to deep learning techniques. [Full paper link(under review)]()

## Dataset Access
- [HR-Extreme Dataset](https://huggingface.co/datasets/NianRan1/HR-Extreme)
- [HR-Extreme Croissant Metadata](https://huggingface.co/api/datasets/NianRan1/HR-Extreme/croissant)

## Index File Access
The code for constructing the dataset is available on GitHub:
- [NOAA Storm Event Database](https://www.ncdc.noaa.gov/stormevents/ftp.jsp)
- [NOAA Storm Prediction Center](https://www.spc.noaa.gov/climo/reports/)

## Dataset Structure
The dataset is organized into two directories:
- `202001_202006`: Data from January 2020 to June 2020
- `202007_202012`: Data from July 2020 to December 2020

Each directory contains the dataset in WebDataset format, following Hugging Face recommendations. Every 10 `.npz` files are aggregated into a single `.tar` file, named sequentially as `i.tar`, where `i` is an integer (e.g., `0001.tar`).

## Usage
To generate a complete index file, use the script `make_datasetall.py` with the start date and end date. For example:
```bash
python make_datasetall.py 20200101 20200630
