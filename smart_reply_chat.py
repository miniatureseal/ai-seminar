from openai_key_retrieval import get_openai_key
from dummy_data_access import DummyDataAccess
from template_generator import TemplateGenerator

import chromadb
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from operator import itemgetter


class IntelligentChat:
    TOP_K_SIM_MESSAGES_RETURNED = 3
    SIMILARITY_THRESHOLD = 0.5

    def __init__(self, chatUser, activeChatId):
        openai_key = get_openai_key()
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        chat_model = ChatOpenAI(openai_api_key=openai_key)
        self.dummy_data_access = DummyDataAccess()
        self.template_generator = TemplateGenerator()

        self.chatUser = chatUser
        self.user_profile = self.dummy_data_access.read_profile(self.chatUser)
        self.context_chats = self.dummy_data_access.read_chats(
            self.user_profile["activeChats"]
        )
        self.active_chat = self.context_chats.pop(activeChatId, None)
        self.chatPartnerUserId = self._getChatPartnerID()
        self.rag_chain = (
            {
                "previous_chat_context": itemgetter("message_to_reply_to")
                | self._initRetiever()
                | self._format_docs,
                "activeUserId": itemgetter("activeUserId"),
                "chat_history": itemgetter("chat_history"),
                "chatPartnerUserId": itemgetter("chatPartnerUserId"),
            }
            | self._generate_prompt()
            | chat_model
            | StrOutputParser()
        )

    def add_message(self, message):
        new_message = {
            "senderUserId": self.chatUser,
            "content": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.active_chat["messages"].append(new_message)
        self.vectorstore.add_documents([new_message])

    def _getChatPartnerID(self):
        participating_user = self.active_chat["participatingUsers"]

        if "amueller" in participating_user:
            participating_user.remove(self.chatUser)

        return participating_user[0]

    def generate_smart_replies(self):
        return self.rag_chain.invoke(
            {
                "message_to_reply_to": self._get_message_to_reply_to(),
                "chat_history": self.active_chat["messages"],
                "activeUserId": self.chatUser,
                "chatPartnerUserId": self.chatPartnerUserId,
            }
        )

    def _format_docs(self, docs):
        return [doc.page_content for doc in docs]

    def _get_message_to_reply_to(self):
        message = max(
            (
                message
                for message in self.active_chat["messages"]
                if message["senderUserId"] != self.chatUser
            ),
            key=lambda x: x["timestamp"],
            default=None,
        )
        return message["content"]

    def _generate_prompt(self):
        # Even tho both prompt could take the same parameters, I separated them
        # so that in the future different context for each of the roles can get
        # retrieved, such as the job description for the company
        if self.user_profile["role"] == "company":
            return self.template_generator.get_company_prompt_template()

        if self.user_profile["role"] == "job_seeker":
            return self.template_generator.get_job_seeker_prompt_template()

    def _initRetiever(self):
        vectorstore_client = chromadb.EphemeralClient()
        past_user_chat_docs = []
        for chat_data in self.context_chats.values():
            for message in chat_data["messages"]:
                if message["senderUserId"] == self.chatUser:
                    past_user_chat_docs.append(
                        Document(
                            page_content=message["content"],
                            metadata={"source": "local"},
                        )
                    )

        vectorstore = Chroma.from_documents(
            documents=past_user_chat_docs,
            embedding=self.embeddings,
            client=vectorstore_client,
            collection_name="openai_collection",
        )
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": self.SIMILARITY_THRESHOLD,
                "k": self.TOP_K_SIM_MESSAGES_RETURNED,
            },
        )

    def _persist_chat(self):
        self.dummy_data_access.persist_chat(self.active_chat)
