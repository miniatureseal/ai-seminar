import tkinter as tk
from chat_ui import ChatInterface
from smart_reply_chat import IntelligentChat
from db_helper import ChromaDBHelper
from langchain.schema import OutputParserException


scenarios = {
    "amueller": ["26", "27", "28", "29", "30"],
    "etaylor": ["31", "32", "33", "34", "35"],
    "InnovateTech": ["36", "37", "38", "39", "40"],
    "GreenScape": ["41", "42", "43", "44", "45"],
}

# Scenario which is picked
scenario_user = "amueller"
# Name of the participant of the experiment, used for logging by whom a message was written
experiment_participant_name = "Paul Blöcher"

db_helper = ChromaDBHelper()
db_helper.initialize_db()

root = tk.Tk()
chat_interface = ChatInterface(
    root,
    scenario_user,
    scenarios[scenario_user],
    experiment_participant_name,
)
root.update_idletasks()
root.mainloop()
