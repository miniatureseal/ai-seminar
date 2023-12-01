# Repo for the research seminar: Enhancing efficiency in job matching apps through AI-suggested chat answers based on data from past conversations

This is a repo for a research seminar at TU Darmstadt which analysis how AI based chat reply suggestions are perceived by users.

## How to run the experiment

To run the code required for the experiment you need to follow these steps:

1. Make sure to have python and conda installed on your system. If not, check out [this link to install python](https://www.python.org/downloads/) or [this link to install conda](https://docs.conda.io/projects/conda/en/latest/index.html) is described how you can do it.
2. Clone the repository to the directory of your choice.
3. Now you need to install the required dependencies. You can do that by opening a terminal window on linux or mac os or git bash on windows. Navigate to the root of this repository and run `conda create --name smart-replies --file requirements.txt`. This should install all needed dependencies for running the code.
4. Now, open up the repository in the code editor or IDE of your choice. Open the file `main.py` in the `src` folder and modify the variables `experiment_participant_name`, the `chat_id`, as well as the `chat_user_id`. The documentation for the variables can be found in the code.
5. Now run `main.py` and hand the laptop to the person taking part in the experiment.

## Reseting the database

At some point while experimenting you might want to reset the vectorstore database, for example when you changed the dummy chats in some way. You can do so by deleting all the files in the database folder except the `.gitkeep` file and then just reruning `main.py`. This will automatically instatiate the vectorstore anew.
