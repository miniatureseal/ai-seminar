import tkinter as tk
from chat_ui import ChatInterface
from smart_reply_chat import IntelligentChat
from db_helper import ChromaDBHelper

chat_user_id = "amueller"
chat_id = "2"

db_helper = ChromaDBHelper()
db_helper.initialize_db()

active_chat = IntelligentChat(chat_user_id, chat_id)
smart_replies = active_chat.generate_smart_replies()
print(smart_replies)

root = tk.Tk()
chat_interface = ChatInterface(root, active_chat.get_message_data(), chat_user_id)
chat_interface.populate_suggestions(
    [smart_replies.reply1, smart_replies.reply2, smart_replies.reply3]
)
root.mainloop()
