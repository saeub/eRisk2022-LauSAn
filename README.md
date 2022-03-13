# eRisk2022-LauSAn

## Setup

1. Install dependencies:
   ```bash
   $ pip install -r requirements.txt
   ```
1. Copy data for task 2 into the `data` directory.

## Training

```bash
$ python src/main.py train ModelType
```

Use `python src/main.py train --help` to see available model types.

## Testing

The final submission will work via a [JSON API](https://erisk.irlab.org/server.html), so we use a local dummy API (running at http://localhost:5000) with the same interface to evaluate our results and test our submission client during development.

1. Run the local submission API, providing test XML files as arguments. For example, using cases from the 2017 dataset as the test set:
   ```bash
   $ python src/api.py data/training_t2/TRAINING_DATA/2017_cases/*/*.xml
   ```
1. Run the submission client:
   ```bash
   $ python src/main.py submit path/to/model1 path/to/model2 ...
   ```
1. Go to http://localhost:5000/results to analyze results.
