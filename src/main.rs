use clap::Parser as ArgParser;
use linfa::prelude::*;
use linfa::Dataset;
use linfa_trees::DecisionTree;
use ndarray::{Array, Array1, Array2};
use std::path::{Path, PathBuf};

#[derive(ArgParser)]
#[clap(version, long_about = None)]
struct Args {
    #[clap(value_parser)]
    filename: PathBuf,
}

fn main() -> Result<(), linfa::Error> {
    let args = Args::parse();

    let path = Path::new(&args.filename);
    let dataset = load_csv(path)?;
    //println!("{:?}", dataset);

    let (train, test) = dataset.split_with_ratio(0.9);
    let model = DecisionTree::params().fit(&train).unwrap();
    let prediction = model.predict(&test);
    let cm = prediction.confusion_matrix(&test)?;
    //println!("{:?}", prediction);
    //println!("{:?}", test.targets);
    println!("Accuracy: {:.1}%", 100.0 * cm.accuracy());
    Ok(())
}

//#[allow(dead_code)]
fn load_csv(path: &Path) -> Result<Dataset<f32, usize, ndarray::Dim<[usize; 1]>>, Error> {
    let mut reader = csv::Reader::from_path(path).unwrap();

    let headers: Vec<String> = reader
        .headers()
        .unwrap()
        .iter()
        .map(|r| r.to_owned())
        .collect();
    #[cfg(build = "debug")]
    println!("Headers: {:?}", headers);

    let data = reader
        .records()
        .map(|r| {
            r.unwrap()
                .iter()
                .map(|field| field.parse::<f32>().unwrap())
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();

    // The target index is the last column of the data
    let max_index = headers.len() - 1;

    let features = headers[0..max_index].to_vec();
    #[cfg(build = "debug")]
    println!("Features: {:?}", features);
    #[cfg(build = "debug")]
    println!("Target name: {}", &headers[max_index]);

    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
        records.extend_from_slice(&record[0..max_index]);
    }
    let rows = records.len() / max_index;
    #[cfg(build = "debug")]
    println!("Data size: {}x{}", rows, max_index);
    let records: Array2<f32> = Array::from(records).into_shape((rows, max_index)).unwrap();

    let targets: Array1<usize> = Array::from(
        data.iter()
            .map(|record| record[max_index] as usize)
            .collect::<Vec<usize>>(),
    );

    Ok(Dataset::new(records, targets).with_feature_names(features))
}
