from openai_key_retrieval import get_openai_key
from dummy_data_access import DummyDataAccess
from template_generator import TemplateGenerator
from response_object import ResponseObject
from db_helper import ChromaDBHelper

import configparser
from pyprojroot.here import here
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.globals import set_debug
from operator import itemgetter
from langchain_core.runnables import RunnableLambda


class IntelligentChat:
    """
    Heartpiece of the chatbot. This class is responsible for generating smart replies.
    It takes the chat user and the current active chat id and initializes the smart
    reply langchain. It also acts as the data layer to the chat UI and handles
    the loading of the dummy data for the given chat and chat actors.
    """

    def __init__(self, chat_user, active_chat_id):
        self.config = configparser.ConfigParser()
        self.config.read(here("config.ini"))
        set_debug(self.config.get("SETTINGS", "DEBUG"))
        self.openai_key = get_openai_key()
        chat_model = ChatOpenAI(
            model=self.config.get("OPENAI", "VERSION"),
            openai_api_key=self.openai_key,
            temperature=0.1,
        )
        self.dummy_data_access = DummyDataAccess()
        self.db_helper = ChromaDBHelper()
        self.output_parser = PydanticOutputParser(pydantic_object=ResponseObject)
        self.template_generator = TemplateGenerator(self.output_parser)

        self.chat_user = chat_user
        self.user_profile = self.dummy_data_access.read_profile(self.chat_user)
        self.context_chats = self.dummy_data_access.read_chats(
            self.user_profile["activeChats"]
        )
        self.active_chat_id = active_chat_id
        self.active_chat = self.context_chats.pop(active_chat_id, None)
        self.chat_partner_user_id = self._get_chat_partner_id()

        self.rag_chain = (
            {
                "previous_chat_context": {
                    "message_to_reply_to": itemgetter("message_to_reply_to"),
                    "chat_user": itemgetter("active_user_id"),
                    "active_chat_id": itemgetter("active_chat_id"),
                }
                | RunnableLambda(self._call_retriever)
                | self._format_docs,
                "active_user_id": itemgetter("active_user_id"),
                "chat_history": itemgetter("chat_history"),
                "chat_partner_user_id": itemgetter("chat_partner_user_id"),
                "message_to_reply_to": itemgetter("message_to_reply_to"),
            }
            | self._generate_prompt()
            | chat_model
            | self.output_parser
        )

    def _get_chat_partner_id(self):
        participating_user = self.active_chat["participatingUsers"]

        if self.chat_user in participating_user:
            participating_user.remove(self.chat_user)

        return participating_user[0]

    def generate_smart_replies(self):
        return self.rag_chain.invoke(
            {
                "message_to_reply_to": self._get_messages_to_reply_to(),
                "chat_history": self.active_chat["messages"],
                "active_user_id": self.chat_user,
                "chat_partner_user_id": self.chat_partner_user_id,
                "active_chat_id": self.active_chat_id,
            }
        )

    def get_message_data(self):
        return self.active_chat["messages"]

    def _format_docs(self, docs):
        return [doc[0].page_content for doc in docs]

    def _call_retriever(self, relevant_data):
        results = self.db_helper.similarity_search_with_relevance_scores(
            relevant_data["message_to_reply_to"],
            relevant_data["chat_user"],
            relevant_data["active_chat_id"],
        )
        return results

    def _get_message_to_reply_to(self):
        message = max(
            (
                message
                for message in self.active_chat["messages"]
                if message["senderUserId"] != self.chat_user
            ),
            key=lambda x: x["timestamp"],
            default=None,
        )
        return message["content"]

    def _get_messages_to_reply_to(self):
        latest_messages = []

        for message in reversed(self.active_chat["messages"]):
            sender = message["senderUserId"]

            if sender == self.chat_user:
                break

            if sender == self.chat_partner_user_id:
                latest_messages.append(message["content"])

        if not latest_messages:
            return ""

        if len(latest_messages) == 1:
            return latest_messages[0]

        return " ".join(latest_messages)

    def _generate_prompt(self):
        if self.user_profile["role"] == "work_provider":
            return self.template_generator.get_company_prompt_template()

        if self.user_profile["role"] == "job_seeker":
            return self.template_generator.get_job_seeker_prompt_template()
