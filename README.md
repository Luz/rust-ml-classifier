# rust-ml-classifier
A machine learning example for classification, written in Rust.

## Building and running
```Shell
make
```

## Data in this repo
The csv file in this repo contains heart disease data.
It is a manually modified version of a freely available dataset.
The manual change is to move the 5% first lines (15) to the end of the file.
This ensures test data (last 10%) contains also people with heart disease.
