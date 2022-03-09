import sys

from flask import Flask, jsonify, request

from data import parse_subject

NUM_RUNS = 1
SUBJECTS = {
    subject.id: subject
    for subject in (parse_subject(filename) for filename in sys.argv[1:3])
}
number = 0
runs = [{nick: [] for nick in SUBJECTS} for _ in range(NUM_RUNS)]

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
                "date": subject.posts[number].date.isoformat(" ", "seconds"),
            }
            for subject in SUBJECTS.values()
            if len(subject.posts) > number
        ]
    )


@app.route("/submit/<team_token>/<int:run_number>", methods=["POST"])
def submit(team_token, run_number):
    data = request.get_json()
    assert len(data) == len(
        SUBJECTS
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
        runs[run_number][nick].append(decision)
    global number
    number += 1
    return jsonify(None)


@app.route("/results", methods=["GET"])
def results():
    # TODO: Evaluate and render metrics as HTML
    return jsonify(runs)


if __name__ == "__main__":
    app.run()
