
from typing import Literal
from openai_key_retrieval import get_openai_key
from langchain.chat_models import ChatOpenAI
from datetime import datetime
from template_generator import TemplateGenerator
from dummy_data_loader import DummyDataLoader

class IntelligentChat:

    def __init__(self, chatUser, activeChatId):
        self.chat_model = ChatOpenAI(get_openai_key())
        self.dummy_data_loader = DummyDataLoader()
        self.template_generator = TemplateGenerator()
        self.chatUser = chatUser
        self.user_profile = self._loadUserProfile(chatUser)
        self._load_messages(activeChatId)

    def add_message(self, message):
        new_message = {
            "senderUserId": self.chatUser,
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        self.active_chat["messages"].append(new_message)

    def generate_smart_replies(self):
        # TODO
        return;

    def _generate_prompt(self):
        # Even tho both prompt could take the same parameters, I separated them
        # so that in the future different context for each of the roles can get 
        # retrieved, such as the job description for the company
        if self.user_profile["role"] == "company":
            prompt = self.template_generator.get_company_prompt_template()
            return prompt.format(
                role=self.user_profile["role"],
                chat_history=self.messages,
                company_profile=self.user_profile
            )
        
        if self.user_profile["role"] == "job_seeker":
            prompt = self.template_generator.get_job_seeker_prompt_template()
            return prompt.format(
                role=self.user_profile["role"],
                chat_history=self.messages,
                job_seeker_profile=self.user_profile
            )
        
    def _load_chats(self, activeChatId):
        # We load all user chats as we want to use them as context for the smart reply prompts
        # later on
        chats = self.dummy_data_loader.read_chats(self.user_profile["chatIds"])
        self.active_chat = chats.pop(activeChatId, None)
        self.context_chats = chats
    
    def _persist_chat():
        return
