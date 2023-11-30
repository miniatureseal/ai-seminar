import uuid
from openai_key_retrieval import get_openai_key
from dummy_data_access import DummyDataAccess
from template_generator import TemplateGenerator
from response_object import ResponseObject

import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.globals import set_debug
from operator import itemgetter


class IntelligentChat:
    TOP_K_SIM_MESSAGES_RETURNED = 3
    SIMILARITY_THRESHOLD = 0.6
    DB_PATH = "./database"

    def __init__(self, chat_user, active_chat_id):
        set_debug(True)
        self.openai_key = get_openai_key()
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key)
        chat_model = ChatOpenAI(openai_api_key=self.openai_key)
        self.dummy_data_access = DummyDataAccess()
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
                "previous_chat_context": itemgetter("message_to_reply_to")
                | self._initRetiever()
                | self._format_docs,
                "active_user_id": itemgetter("active_user_id"),
                "chat_history": itemgetter("chat_history"),
                "chat_partner_user_id": itemgetter("chat_partner_user_id"),
            }
            | self._generate_prompt()
            | chat_model
            | self.output_parser
        )

    def add_message(self, message):
        new_message = {
            "senderUserId": self.chat_user,
            "content": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.active_chat["messages"].append(new_message)
        self.vectorstore.add_documents([new_message])

    def _get_chat_partner_id(self):
        participating_user = self.active_chat["participatingUsers"]

        if "amueller" in participating_user:
            participating_user.remove(self.chat_user)

        return participating_user[0]

    def generate_smart_replies(self):
        return self.rag_chain.invoke(
            {
                "message_to_reply_to": self._get_message_to_reply_to(),
                "chat_history": self.active_chat["messages"],
                "active_user_id": self.chat_user,
                "chat_partner_user_id": self.chat_partner_user_id,
            }
        )

    def get_message_data(self):
        return self.active_chat["messages"]

    def _format_docs(self, docs):
        return [doc.page_content for doc in docs]

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

    def _generate_prompt(self):
        # Even thox both prompt could take the same parameters, I separated them
        # so that in the future different context for each of the roles can get
        # retrieved, such as the job description for the company
        if self.user_profile["role"] == "company":
            return self.template_generator.get_company_prompt_template()

        if self.user_profile["role"] == "job_seeker":
            return self.template_generator.get_job_seeker_prompt_template()

    def _initRetiever(self):
        embed_fct = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_key, model_name="text-embedding-ada-002"
        )
        vectorstore_client = chromadb.PersistentClient(path=self.DB_PATH)
        message_collection = vectorstore_client.get_or_create_collection(
            name="chat_messages", embedding_function=embed_fct
        )

        past_user_chats = []
        chat_metadata = []
        for chat_data in self.context_chats.values():
            for message in chat_data["messages"]:
                if message["senderUserId"] == self.chat_user:
                    past_user_chats.append(message["content"])
                    chat_metadata.append(
                        {
                            "user": self.chat_user,
                            "chat_id": chat_data["chatId"],
                        }
                    )

        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc)) for doc in past_user_chats]
        unique_ids = list(set(ids))

        seen_ids = set()
        unique_docs = [
            doc
            for doc, id in zip(past_user_chats, ids)
            if id not in seen_ids and (seen_ids.add(id) or True)
        ]

        message_collection.add(
            ids=unique_ids, documents=unique_docs, metadatas=chat_metadata
        )

        langchain_chroma = Chroma(
            client=vectorstore_client,
            collection_name="chat_messages",
            embedding_function=self.embeddings,
        )

        filter = {
            "$and": [
                {"user": {"$eq": self.chat_user}},
                {"chat_id": {"$ne": int(self.active_chat_id)}},
            ]
        }

        return langchain_chroma.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": self.SIMILARITY_THRESHOLD,
                "k": self.TOP_K_SIM_MESSAGES_RETURNED,
                "filter": filter,
            },
        )

    def _persist_chat(self):
        self.dummy_data_access.persist_chat(self.active_chat)
