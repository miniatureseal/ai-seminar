import json
import os
import tkinter as tk
from tkinter import scrolledtext
from pyprojroot.here import here
from nltk import download
from nltk.translate import bleu_score
from nltk.tokenize import word_tokenize

from smart_reply_chat import IntelligentChat
from langchain.schema import OutputParserException


class ChatInterface:
    def __init__(
        self,
        root,
        active_user,
        chat_ids,
        experiment_participant_name,
    ):
        download("punkt")
        self.experiment_participant_name = experiment_participant_name.replace(
            " ", ""
        ).lower()
        self.last_selected_suggestion = ""
        self.current_chat_index = 0
        self.active_user = active_user
        self.chat_ids = chat_ids

        self.root = root
        self.root.title("Smart reply chat")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"{screen_width}x{screen_height}")

        self.chat_display = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=80, height=30
        )
        self.chat_display.pack(padx=10, pady=10)

        self.suggestion_texts = []
        for i in range(3):
            text = tk.Text(root, wrap=tk.WORD, height=5, width=23)
            text.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            text.bind(
                "<ButtonRelease-1>",
                lambda event, index=i: self._insert_suggestion(event),
            )
            self.suggestion_texts.append(text)

        divider = tk.Frame(root, height=2, width=70, bd=1, relief=tk.SUNKEN)
        divider.pack(pady=1)

        self.entry_field = tk.Text(root, wrap=tk.WORD, height=5, width=70)
        self.entry_field.pack(pady=5)

        self.send_button = tk.Button(root, text="Send", command=self._send_message)
        self.send_button.pack(pady=5)
        self._load_chats_from_index()

    def _load_chats_from_index(self):
        self._reset_inputs()

        active_chat = IntelligentChat(
            self.active_user, self.chat_ids[self.current_chat_index]
        )
        self._display_messages(active_chat.get_message_data(), self.active_user)
        try:
            smart_replies = active_chat.generate_smart_replies()
            self._populate_suggestions(
                [smart_replies.reply1, smart_replies.reply2, smart_replies.reply3]
            )
        except OutputParserException:
            self._populate_suggestions(["NA"] * 3)

    def _reset_inputs(self):
        self.chat_display.delete("1.0", tk.END)

        for i in range(3):
            suggestion = self.suggestion_texts[i]
            suggestion.delete("1.0", tk.END)

    def _send_message(self):
        message = self.entry_field.get("1.0", tk.END).strip()

        self._display_message_for_active_user(message)
        suggested_tokens = word_tokenize(self.last_selected_suggestion.lower())
        user_edited_tokens = word_tokenize(message.lower())
        bleu_result = bleu_score.sentence_bleu(
            [suggested_tokens], user_edited_tokens, weights=(1, 0, 0, 0)
        )
        self._save_user_input_to_file(
            {
                "written_by": self.experiment_participant_name,
                "chat_id": self.chat_ids[self.current_chat_index],
                "chosen_suggestion": self.last_selected_suggestion,
                "written_message": message,
                "bleu_score": bleu_result,
            }
        )

        self.last_selected_suggestion = ""
        self.current_chat_index += 1
        self.entry_field.delete("1.0", tk.END)
        if self.current_chat_index < len(self.chat_ids):
            self._load_chats_from_index()
        else:
            self.send_button.config(state="disabled")
            self.entry_field.insert(
                tk.END, "Thank you for taking part in the experiment!"
            )

    def _populate_suggestions(self, suggestions):
        if len(suggestions) == len(self.suggestion_texts):
            for i, suggestion in enumerate(suggestions):
                text = self.suggestion_texts[i]
                text.insert(tk.END, suggestion)

    def _insert_suggestion(self, event):
        selected_suggestion = event.widget.get("1.0", tk.END).strip()
        self.last_selected_suggestion = selected_suggestion
        self.entry_field.delete("1.0", tk.END)
        self.entry_field.insert(tk.END, selected_suggestion)

    def _display_messages(self, messages, active_user):
        for message in messages:
            sender = message["senderUserId"]
            content = message["content"]
            self._display_message(sender, content, active_user)

    def _display_message_for_active_user(self, content):
        self._display_message(self.active_user, content, self.active_user)

    def _display_message(self, sender, content, active_user):
        sender = active_user + "(You)" if sender == active_user else sender
        formatted_message = f"{sender}: {content}"
        self.chat_display.insert(tk.END, formatted_message + "\n\n")

    def _save_user_input_to_file(self, data):
        folder_path = str(here("src/output/" + self.experiment_participant_name))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = f"participant_{self.experiment_participant_name}_chatId_{self.chat_ids[self.current_chat_index]}.json"
        filepath = str(here(folder_path + "/" + filename))
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
