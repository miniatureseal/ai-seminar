import tkinter as tk
from chat_ui import ChatInterface
from smart_reply_chat import IntelligentChat
from db_helper import ChromaDBHelper

# Chat user from which perspective the chat is opened
chat_user_id = "amueller"
# Chat in which user with user id chat_user_id is participating in
chat_id = "1"
# Name of the participant of the experiment, used for logging by whom a message was written
experiment_participant_name = "Jan"

db_helper = ChromaDBHelper()
db_helper.initialize_db()

active_chat = IntelligentChat(chat_user_id, chat_id)
smart_replies = active_chat.generate_smart_replies()
print(smart_replies)

root = tk.Tk()
chat_interface = ChatInterface(
    root,
    active_chat.get_message_data(),
    chat_user_id,
    chat_id,
    experiment_participant_name,
)
chat_interface.populate_suggestions(
    [smart_replies.reply1, smart_replies.reply2, smart_replies.reply3]
)
root.mainloop()
