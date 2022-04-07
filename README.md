# eRisk2022-LauSAn

## Setup

1. Install dependencies:
   ```bash
   $ pip install -r requirements.txt
   ```
1. Copy data for task 2 into the `data` directory.

## Training

We use two different train/test sets for evaluating our models before submission. To train on set 1:

```bash
$ python src/main.py train ModelType --data data/train_set_1.txt
```

By default, the model will be saved with a timestamp in the filename in the current working directory. Use `python src/main.py train --help` to see available model types and other options.

To optimize the threshold scheduler parameters, use:

```bash
$ python src/main.py optimize-threshold my_model.pickle --metric erde5 --data data/train_set_1.txt
```

This will perform grid search over a range of parameter values defined in the model class and store the optimized model in a second file (by default, appending `.optimized` to the filename base).

To see information about a saved model, including optimized threshold scheduler parameters, use `python src/main.py info my_model.pickle`.

## Testing

The final submission will work via a [JSON API](https://erisk.irlab.org/server.html), so we use a local dummy API (running at http://localhost:5000) with the same interface to evaluate our results and test our submission client during development.

1. Run the local submission API, providing the test set and the number of runs as an argument. For example, using our test set 1:
   ```bash
   $ python src/api.py --data data/test_set_1.txt --runs 2
   ```
1. Run the submission client (one model per run):
   ```bash
   $ python src/main.py submit path/to/model1 path/to/model2 ...
   ```
1. Go to http://localhost:5000/results to analyze results.
