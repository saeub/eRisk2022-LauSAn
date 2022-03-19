import argparse
from datetime import datetime

import dateutil.parser
import requests
from tqdm import tqdm

import models
from data import Post, Subject, parse_subject


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
    train_parser.add_argument("subjects", nargs="*", help="Training subject XML files.")
    train_parser.add_argument(
        "--save-path", help="File name for storing the trained model."
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

    args = parser.parse_args()
    if args.command == "train":
        args.model_class = model_classes[args.model_class]
    return args


def train(args):
    model: models.Model = args.model_class()
    subjects = [parse_subject(filename) for filename in args.subjects]
    model.train(subjects)
    save_path = (
        args.save_path
        or f"{args.model_class.__name__}_{datetime.now().isoformat()}.pickle"
    )
    models.save(model, save_path)


def submit(args):
    run_models = [models.load(model) for model in args.models]

    subjects = {}

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
            subject.posts.append(
                Post(
                    writing["title"],
                    dateutil.parser.isoparse(writing["date"]),
                    writing["content"],
                )
            )

        # Predict and post results
        for run, model in enumerate(run_models):
            predictions = model.predict(
                [subjects[subject_id] for subject_id in subject_ids]
            )
            data = [
                {
                    "nick": subject_id,
                    "decision": 0 if prediction < 0.5 else 1,
                    "score": prediction,
                }
                for subject_id, prediction in zip(subject_ids, predictions)
            ]
            requests.post(f"{args.api}/submit/{args.team_token}/{run}", json=data)


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "submit":
        submit(args)
    else:
        raise NotImplementedError(f"Command {args.command} not implemented")
