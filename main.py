from smart_reply_chat import IntelligentChat

active_chat = IntelligentChat("amueller", "1")
smart_reply = active_chat.generate_smart_replies()
print(smart_reply)
