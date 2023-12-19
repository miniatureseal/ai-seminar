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
scenario_user = "GreenScape"
# Name of the participant of the experiment, used for logging by whom a message was written
experiment_participant_name = "Paul Blöcher"

db_helper = ChromaDBHelper()
db_helper.initialize_db()


for chat_id in scenarios[scenario_user]:
    active_chat = IntelligentChat(scenario_user, chat_id)
    root = tk.Tk()
    chat_interface = ChatInterface(
        root,
        active_chat.get_message_data(),
        scenario_user,
        chat_id,
        experiment_participant_name,
    )
    try:
        smart_replies = active_chat.generate_smart_replies()
        chat_interface.populate_suggestions(
            [smart_replies.reply1, smart_replies.reply2, smart_replies.reply3]
        )
    except OutputParserException:
        chat_interface.populate_suggestions(["NA"] * 3)

    root.mainloop()
