import argparse

from flask import Flask, jsonify, request

import data
import evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("subjects", nargs="*", help="Test subject XML files.")
    parser.add_argument("--port", type=int, default=5000, help="Server port.")
    return parser.parse_args()


args = parse_args()

NUM_RUNS = 1
SUBJECTS = {
    subject.id: subject
    for subject in (data.parse_subject(filename) for filename in args.subjects)
}
number = 0
runs = [{subject_id: [] for subject_id in SUBJECTS} for _ in range(NUM_RUNS)]

app = Flask(__name__)


@app.route("/getwritings/<team_token>", methods=["GET"])
def getwritings(team_token):
    return jsonify(
        [
            {
                "id": 12345,
                "number": number,
                "nick": subject.id,
                "redditor": 12345,
                "title": subject.posts[number].title,
                "content": subject.posts[number].text,
                "date": subject.posts[number].date.isoformat(),
            }
            for subject in SUBJECTS.values()
            if len(subject.posts) > number
        ]
    )


@app.route("/submit/<team_token>/<int:run_number>", methods=["POST"])
def submit(team_token, run_number):
    data = request.get_json()
    assert len(data) == len(
        [subject for subject in SUBJECTS.values() if len(subject.posts) > number]
    ), f"Sent decisions for {len(data)} subjects instead of {len(SUBJECTS)}"
    for d in data:
        nick = d["nick"]
        decision = d["decision"]
        score = d["score"]
        assert nick in SUBJECTS, f"Inexistent subject ID {nick}"
        assert decision in [0, 1], f"Invalid decision {decision}, must be 0 or 1"
        assert isinstance(
            score, float
        ), f"Score has type {type(score)}, should be a float"
        runs[run_number][nick].append((bool(decision), score))
    global number
    number += 1
    return jsonify(None)


@app.route("/results", methods=["GET"])
def results():
    runs_html = ""
    for run_number, run in enumerate(runs):
        decisions = {
            subject_id: [decision for (decision, score) in predictions]
            for subject_id, predictions in run.items()
        }
        erde_5 = evaluation.mean_erde(decisions, SUBJECTS.values(), o=5)
        erde_50 = evaluation.mean_erde(decisions, SUBJECTS.values(), o=50)
        recall, precision, f1 = evaluation.recall_precision_f1(
            decisions, SUBJECTS.values()
        )
        metrics_html = f"""
            <ul>
                <li><i>ERDE<sub>5</sub></i> = {erde_5}</li>
                <li><i>ERDE<sub>50</sub></i> = {erde_50}</li>
                <li><i>R</i> = {recall}</li>
                <li><i>P</i> = {precision}</li>
                <li><i>F<sub>1</sub></i> = {f1}</li>
            </ul>
        """

        runs_html += f"""
            <h2><a href="/results/{run_number}">Run {run_number}</a></h2>
            {metrics_html}
        """

    return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Evaluation results</title>
            </head>
            <body style="font-family: sans-serif">
                <h1>Evaluation results</h1>
                {runs_html}
            </body>
        </html>
    """


@app.route("/results/<int:run_number>", methods=["GET"])
def results_run(run_number):
    run = runs[run_number]
    max_num_posts = max(len(subject.posts) for subject in SUBJECTS.values())
    rows_html = f"""
        <tr>
            <th>Subject</th>
            <th>True label</th>
            {"".join(f'<th>{i}</th>' for i in range(max_num_posts))}
        </tr>
    """
    for subject_id, subject in SUBJECTS.items():
        cells_html = ""
        decision_made = False
        for i, (decision, score) in enumerate(run[subject_id]):
            post_link = f"/posts/{subject_id}#{i}"
            if decision_made:
                cells_html += f"""
                    <td>
                        <a href="{post_link}"
                           target="_blank"
                           title="{score}"
                           style="color: lightgray">{int(decision)}</a>
                    </td>
                """
            else:
                cells_html += f"""
                    <td>
                        <a href="{post_link}"
                           target="_blank"
                           title="{score}">{int(decision)}</a>
                    </td>
                """
            if decision == 1:
                decision_made = True
        rows_html += f"""
            <tr>
                <td>{subject_id}</td>
                <td>{int(subject.label)}</td>
                {cells_html}
            </tr>
        """

    return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Evaluation results</title>
            </head>
            <body style="font-family: sans-serif">
                <h1>Run {run_number}</h2>
                <table>
                    {rows_html}
                </table>
            </body>
        </html>
    """


@app.route("/posts/<subject_id>", methods=["GET"])
def posts(subject_id):
    if subject_id not in SUBJECTS:
        return f"Subject with ID {subject_id} not found", 404
    subject = SUBJECTS[subject_id]
    posts_html = ""
    for i, post in enumerate(subject.posts):
        posts_html += f"""
            <h2 id="{i}">{post.title or "<i>(no title)</i>"}</h2>
            <p><i>{post.date}</i></p>
            <p>{post.text}</p>
            <hr>
        """
    return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Posts by {subject.id}</title>
            </head>
            <body style="font-family: sans-serif">
                <h1>Posts by {subject.id}</h1>
                {posts_html}
            </body>
        </html>
    """


app.run(port=args.port)
