from ml_runner.core.utils.path_utils import resolve_path_template
import datetime

def test_resolve_path_template_basic():
    template = "results/{experiment_name}/output.log"
    context = {"experiment_name": "my_exp"}
    resolved = resolve_path_template(template, context)
    assert resolved == "results/my_exp/output.log"

def test_resolve_path_template_date():
    template = "runs/{date:%Y}/job.txt"
    resolved = resolve_path_template(template, {})
    now = datetime.datetime.now()
    expected = f"runs/{now.strftime('%Y')}/job.txt"
    assert resolved == expected

def test_resolve_path_template_complex():
    template = "{date:%d-%m-%Y}/{experiment_name}/epoch_{epoch}.pt"
    context = {"experiment_name": "mnist", "epoch": 5}
    resolved = resolve_path_template(template, context)
    now = datetime.datetime.now()
    expected = f"{now.strftime('%d-%m-%Y')}/mnist/epoch_5.pt"
    assert resolved == expected
