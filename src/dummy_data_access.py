import json
import os
from pyprojroot.here import here


class DummyDataAccess:
    """
    Simple class to read the chat and profile dummy data from the file system.
    Dummy data can be found in the folder src/dummy_data.
    """

    def __init__(self):
        self.chat_folder_path = str(here("src/dummy_data/chats"))
        self.company_profile_path = str(here("src/dummy_data/company_profiles"))
        self.job_seeker_profile_path = str(here("src/dummy_data/job_seeker_profiles"))

    def read_chats(self, chat_ids):
        chats_dict = {}

        for chat_id in chat_ids:
            chat_file_path = os.path.join(self.chat_folder_path, f"{chat_id}.json")

            chat_data = self._read_json(chat_file_path)
            if chat_data != None:
                chats_dict[chat_id] = chat_data
            else:
                print(f"Chat data not found for {chat_id}")

        return chats_dict

    def read_single_chat(self, chat_id):
        chat_file_path = os.path.join(self.chat_folder_path, f"{chat_id}.json")
        chat_data = self._read_json(chat_file_path)
        if chat_data == None:
            print(f"Chat data not found for {chat_id}")
        return chat_data

    def read_chat_data(self):
        chat_data = []

        for filename in os.listdir(self.chat_folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.chat_folder_path, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    chat_data.append(data)

        return chat_data

    def read_profile(self, user_id: str):
        company_profile_path = os.path.join(
            self.company_profile_path, f"{user_id}.json"
        )
        if os.path.exists(company_profile_path):
            with open(company_profile_path, "r", encoding="utf-8") as file:
                company_profile = json.load(file)
            return company_profile

        job_seeker_profile_path = os.path.join(
            self.job_seeker_profile_path, f"{user_id}.json"
        )
        if os.path.exists(job_seeker_profile_path):
            with open(job_seeker_profile_path, "r", encoding="utf-8") as file:
                job_seeker_profile = json.load(file)
            return job_seeker_profile

        print(f"Profile not found for user ID {user_id}")
        return None

    def _read_json(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data
        else:
            return None
