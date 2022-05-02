import argparse
import atexit
import csv
import json
import random
import re
import sys
from datetime import datetime
from textwrap import dedent
from typing import Collection
import dataclasses

import dateutil.parser
import numpy
import requests
import torch
from tqdm import tqdm

import evaluation
import models
from data import Post, Subject, parse_subject, undersample_subjects
from log import logger

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

SUBJECTS_JSON_FILENAME = f"subjects_{datetime.now().isoformat()}.json"


class SubjectJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Subject):
            return dataclasses.asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _dump_subjects(subjects: Collection[Subject]):
    with open(SUBJECTS_JSON_FILENAME, "w") as f:
        json.dump(
            [dataclasses.asdict(subject) for subject in subjects],
            f,
            cls=SubjectJSONEncoder,
        )


def parse_args() -> argparse.Namespace:
    parser_kwargs = {"formatter_class": argparse.ArgumentDefaultsHelpFormatter}
    parser = argparse.ArgumentParser(**parser_kwargs)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    model_classes = {model.__name__: model for model in models.Model.__subclasses__()}
    train_parser = subparsers.add_parser("train", **parser_kwargs)
    train_parser.add_argument(
        "model_class", choices=model_classes, help="Type of model."
    )
    train_parser.add_argument(
        "--args",
        type=json.loads,
        default={},
        help="Constructor arguments to use for instantiating model.",
    )
    train_parser.add_argument(
        "--data",
        type=argparse.FileType("r"),
        required=True,
        help="Text file containing paths to training subject XML files.",
    )
    train_parser.add_argument(
        "--undersample",
        type=float,
        help="Undersample negative subjects towards this ratio (negative to positive).",
    )
    train_parser.add_argument(
        "--save-path", help="File name for storing the trained model."
    )

    optimize_threshold_parser = subparsers.add_parser(
        "optimize-threshold", **parser_kwargs
    )
    optimize_threshold_parser.add_argument("model", help="Path to saved model.")
    optimize_threshold_parser.add_argument(
        "--data",
        type=argparse.FileType("r"),
        required=True,
        help="Text file containing paths to training subject XML files.",
    )
    optimize_threshold_parser.add_argument(
        "--metrics",
        choices=evaluation.METRICS,
        nargs="+",
        default=["erde5"],
        help="Metrics to optimize for.",
    )
    optimize_threshold_parser.add_argument(
        "--sample",
        type=int,
        help="Number of sampled subjects to use.",
    )
    optimize_threshold_parser.add_argument(
        "--save-path", help="File name for storing the optimized model."
    )

    submit_parser = subparsers.add_parser("submit", **parser_kwargs)
    submit_parser.add_argument(
        "models", nargs="+", help="Paths to saved model (each model is a separate run)."
    )
    submit_parser.add_argument(
        "--api", default="http://localhost:5000", help="Submission API endpoint."
    )
    submit_parser.add_argument(
        "--team-token", default="dummy_token", help="Team token for submission API."
    )
    submit_parser.add_argument(
        "--resume", help="JSON file with subjects to resume from in case of a crash."
    )

    info_parser = subparsers.add_parser("info", **parser_kwargs)
    info_parser.add_argument("model", help="Path to saved model.")
    info_parser.add_argument(
        "--threshold",
        action="store_true",
        help="Print only results from threshold scheduler grid search in CSV format.",
    )

    args = parser.parse_args()
    if args.command == "train":
        args.model_class = model_classes[args.model_class]
    return args


def train(args):
    model: models.Model = args.model_class(**args.args)

    logger.info("Loading data...")
    subjects = [parse_subject(filename.strip()) for filename in args.data]
    random.shuffle(subjects)
    if args.undersample is not None:
        subjects = undersample_subjects(subjects, args.undersample)

    logger.info("Training model...")
    model.train(subjects)
    save_path = (
        args.save_path
        or f"{args.model_class.__name__}_{datetime.now().isoformat()}.pickle"
    )

    logger.info(f"Saving model to {save_path}...")
    models.save(model, save_path)


def optimize_threshold(args):
    logger.info("Loading model...")
    model = models.load(args.model)

    logger.info("Loading data...")
    subjects = [parse_subject(filename.strip()) for filename in args.data]

    logger.info("Optimizing threshold scheduler...")
    run = None
    for metric_name in args.metrics:
        metric, minimize = evaluation.METRICS[metric_name]
        run = model.optimize_threshold_scheduler(subjects, metric, minimize, run=run)
        save_path = (
            args.save_path
            or re.sub(r".pickle$", "", args.model) + f".optimized_{metric_name}.pickle"
        )
        logger.info(f"Saving model to {save_path}...")
        models.save(model, save_path)


def submit(args):
    run_models = [models.load(model) for model in args.models]

    if args.resume is not None:
        with open(args.resume) as f:
            saved_subjects = json.load(f)
        subjects = {
            subject["id"]: Subject(
                subject["id"],
                [
                    Post(
                        post["title"],
                        dateutil.parser.isoparse(post["date"]),
                        post["text"],
                    )
                    for post in subject["posts"]
                ],
                None,
            )
            for subject in saved_subjects
        }

    @atexit.register
    def dump_subjects():
        _dump_subjects(subjects.values())

    progress = None
    while True:
        # Get new posts
        writings = requests.get(f"{args.api}/getwritings/{args.team_token}").json()
        subject_ids = [writing["nick"] for writing in writings]
        if progress is None:
            progress = tqdm(total=len(subject_ids))
        progress.n = progress.total - len(writings)
        progress.refresh()
        if len(writings) == 0:
            break

        # Add new posts to `subjects`
        for writing in writings:
            subject_id = writing["nick"]
            subject = subjects.setdefault(subject_id, Subject(subject_id, [], None))
            assert writing["number"] == len(subject.posts), (
                f"Internal number of posts by subject '{subject.id}' ({len(subject.posts)}) "
                f"does not agree with API ({writing['number']}). "
                "Was the submission stopped and resumed without importing previous post histories?"
            )
            subject.posts.append(
                Post(
                    writing["title"].strip(),
                    dateutil.parser.isoparse(writing["date"].strip()),
                    writing["content"].strip(),
                )
            )

        # Predict and post results
        for run, model in enumerate(run_models):
            decisions = model.decide(
                [subjects[subject_id] for subject_id in subject_ids]
            )
            data = [
                {
                    "nick": subject_id,
                    "decision": int(decision),
                    "score": score,
                }
                for subject_id, (decision, score) in zip(subject_ids, decisions)
            ] + [
                # Decide "false" for subjects which have no more posts,
                # as the API expects a decision for every subject every time
                {
                    "nick": subject_id,
                    "decision": 0,
                    "score": 0.0,
                }
                for subject_id in subjects
                if subject_id not in subject_ids
            ]
            requests.post(f"{args.api}/submit/{args.team_token}/{run}", json=data)


def info(args):
    model = models.load(args.model)
    if args.threshold:
        if model.threshold_scheduler.grid_search_results:
            grid_search_results = model.threshold_scheduler.grid_search_results
            writer = csv.DictWriter(sys.stdout, [*grid_search_results[0][0], "metric"])
            writer.writeheader()
            for attr_values, result in grid_search_results:
                writer.writerow({**attr_values, "metric": result})
        else:
            print("This model's threshold scheduler is not optimized.", sys.stderr)
            exit(1)
    else:
        print(
            dedent(
                f"""\
                    Type: {model.__class__.__name__}
                    Threshold scheduler: {model.threshold_scheduler!r}
                """.rstrip()
            )
        )


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "optimize-threshold":
        optimize_threshold(args)
    elif args.command == "submit":
        submit(args)
    elif args.command == "info":
        info(args)
    else:
        raise NotImplementedError(f"Command {args.command} not implemented")
