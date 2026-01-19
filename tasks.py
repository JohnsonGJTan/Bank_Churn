from invoke.tasks import task

@task
def clean(c):
    # Clean data
    c.run("rm -rf data/raw/*")
    c.run("rm -rf data/clean/*")

    # Clean pipelines
    c.run("rm -rf pipelines/*")

    # Reset notebooks
    c.run("jupyter nbconvert --clear-output --inplace notebooks/0.data_exploration.ipynb")
    c.run("jupyter nbconvert --clear-output --inplace notebooks/1.churn_predict_binary.ipynb")
    c.run("jupyter nbconvert --clear-output --inplace notebooks/2.high_value_segments.ipynb")

@task
def build_data(c):
    print("Building data...")
    c.run("python bin/build_data.py --raw-dir-path data/raw/ --clean-dir-path data/clean/")

@task(pre=[build_data])
def run_data_nb(c):
    print("Running EDA notebook...")
    c.run("jupyter nbconvert --to notebook --execute --inplace notebooks/0.data_exploration.ipynb")

@task(pre=[build_data])
def run_model_nb(c):
    print("Running model development notebook...")
    c.run("jupyter nbconvert --to notebook --execute --inplace notebooks/1.churn_predict_binary.ipynb")

@task(pre=[build_data])
def build_pipeline(c):
    print("Building predictive model pipeline...")
    c.run("python bin/build_pipeline.py --clean-data-path data/clean/train.csv --pipeline-save-path pipelines/")

@task(pre=[build_pipeline])
def run_value_nb(c):
    print("Running customer value nb")
    c.run("jupyter nbconvert --to notebook --execute --inplace notebooks/2.high_value_segments.ipynb")

@task(pre=[clean, run_data_nb, run_model_nb, run_value_nb])
def reproduce(c):
    print("Reproducing workspace")
    