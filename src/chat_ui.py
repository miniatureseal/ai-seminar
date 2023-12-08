import tkinter as tk
from tkinter import scrolledtext, RIGHT, LEFT, END

import json
import uuid
from pyprojroot.here import here
from nltk import download
from nltk.translate import bleu_score
from nltk.tokenize import word_tokenize


class ChatInterface:
    def __init__(
        self, root, messages, active_user, chat_id, experiment_participant_name
    ):
        download("punkt")
        self.root = root
        self.experiment_participant_name = experiment_participant_name

        self.chat_id = chat_id
        self.root.title("Smart reply chat")
        self.root.geometry("600x800")

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
                lambda event, index=i: self.insert_suggestion(event, index),
            )
            self.suggestion_texts.append(text)

        divider = tk.Frame(root, height=2, width=70, bd=1, relief=tk.SUNKEN)
        divider.pack(pady=1)

        self.entry_field = tk.Text(root, wrap=tk.WORD, height=5, width=70)
        self.entry_field.pack(pady=5)

        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(pady=5)

        self.display_messages(messages, active_user)

    def send_message(self):
        message = self.entry_field.get("1.0", tk.END).strip()

        self.display_user_message("You", "You", message, align=RIGHT)
        suggested_tokens = word_tokenize(self.last_selected_suggestion.lower())
        user_edited_tokens = word_tokenize(message.lower())
        bleu_result = bleu_score.sentence_bleu(
            [suggested_tokens], user_edited_tokens, weights=(1, 0, 0, 0)
        )
        self.save_user_input_to_file(
            {
                "written_by": self.experiment_participant_name,
                "chat_id": self.chat_id,
                "chosen_suggestion": self.last_selected_suggestion,
                "written_message": message,
                "bleu_score": bleu_result,
            }
        )

        self.send_button.config(state="disabled")
        self.entry_field.delete("1.0", tk.END)
        self.entry_field.insert(tk.END, "Thank you for taking part in the experiment!")

    def populate_suggestions(self, suggestions):
        if len(suggestions) == len(self.suggestion_texts):
            for i, suggestion in enumerate(suggestions):
                text = self.suggestion_texts[i]
                text.delete("1.0", tk.END)
                text.insert(tk.END, suggestion)

    def insert_suggestion(self, event, suggestion_number):
        selected_suggestion = event.widget.get("1.0", tk.END).strip()

        self.last_selected_suggestion = selected_suggestion

        self.entry_field.delete("1.0", tk.END)
        self.entry_field.insert(tk.END, selected_suggestion)

    def display_messages(self, messages, active_user):
        for message in messages:
            sender = message["senderUserId"]
            content = message["content"]
            align = RIGHT if sender == active_user else LEFT
            self.display_user_message(sender, active_user, content, align)

    def display_user_message(self, sender, active_user, content, align):
        sender = "You" if sender == active_user else sender
        formatted_message = f"{sender}: {content}"
        self.chat_display.insert(tk.END, formatted_message + "\n\n", align)
        self.chat_display.tag_configure(sender, background="#E0E0E0")

    def save_user_input_to_file(self, data):
        unique_id = str(uuid.uuid4())

        filename = f"{unique_id}.json"
        filepath = str(here("src/output/" + filename))
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)


# Dummy code for testing out the UI
if __name__ == "__main__":
    messages_data = [
        {
            "senderUserId": "amueller",
            "content": "Hello, I'm interested in the job.",
            "timestamp": "2023-01-15T14:30:00",
        },
        {
            "senderUserId": "company",
            "content": "Hello, tell me about your experience.",
            "timestamp": "2023-01-15T14:35:00",
        },
        {
            "senderUserId": "amueller",
            "content": "I have a Master's in Biology.",
            "timestamp": "2023-01-15T14:40:00",
        },
        {
            "senderUserId": "company",
            "content": "Impressive! What are you passionate about?",
            "timestamp": "2023-01-15T14:45:00",
        },
    ]

    active_user = "amueller"

    suggestions = [
        "I'm very passionate about applying molecular biology techniques to advance drug discovery and development. The intersection of research and practical applications is highly rewarding for me.",
        "I have experience in data analysis using Python and R for my research projects during my Master's program. I would love to contribute to your research initiatives.",
        "In my previous position at the Pharmaceutical Institute, I successfully led a team in developing a new assay for drug discovery. We collaborated closely with researchers and delivered the project ahead of schedule.",
    ]

    root = tk.Tk()
    chat_interface = ChatInterface(root, messages_data, active_user)
    chat_interface.populate_suggestions(suggestions)
    root.mainloop()
