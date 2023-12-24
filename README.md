# Repo for the research seminar: Enhancing efficiency in job matching apps through AI-suggested chat answers based on data from past conversations

This is a repo for a research seminar at TU Darmstadt which analysis how AI based chat reply suggestions are perceived by users.

## How to run the experiment

To run the code required for the experiment you need to follow these steps:

1. Make sure to have python and conda installed on your system. If not, check out [this link to install python](https://www.python.org/downloads/) and [this link to install conda](https://docs.conda.io/projects/conda/en/latest/index.html) is described how you can do it.
2. Clone the repository to the directory of your choice.
3. Now you need to install the required dependencies. You can do that by opening a terminal window on linux or mac os or git bash on windows. Navigate to the root of this repository and run `conda env create -f environment.yml`. This should install all needed dependencies for running the code. Switch to the now installed virtual conda environment using `conda activate smart-reply-env`
4. Add your OpenAI API key in the file `config.ini` in the root directory beneath the variable `KEY`. You don't need to add quotation marks.
5. Now open the file `main.py` in the `src` folder and modify the variables `experiment_participant_name`, and the `scenario` to choose which persona and chats to run.
6. Now run `main.py` and hand the laptop to the person taking part in the experiment.

## Other useful commands

### Resetting the vectorstore

At some point while experimenting you might want to reset the vectorstore database, for example when you changed the dummy chats in some way or you want to experiment with the chunk size of the embedded text. You can do so by deleting all the files in the database folder using by running `make reset_vectorstore` in the project root directory. The vectorstore is automatically reinstantiated using the new data when reruning `main.py`.

### Deleting the generated output

When testing the code you may generate output files which you want to delete again. You can reset the output folder to a clean state using `make reset_output`.

### Resetting both the vectorstore and output

You can run `make reset` to run both of the above mentioned commands. Ideal to reset the project to a clean slate.
