import configparser
from openai_key_retrieval import get_openai_key

import chromadb
import uuid
from chromadb.utils import embedding_functions
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dummy_data_access import DummyDataAccess
from pyprojroot.here import here


class ChromaDBHelper:
    """
    This class serves as an interface to load and reset the dummy test data!
    """

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(here("config.ini"))
        self.dummy_data_access = DummyDataAccess()
        self.openai_key = get_openai_key()
        self.embed_fct = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_key, model_name="text-embedding-ada-002"
        )
        self.vectorstore_client = chromadb.PersistentClient(
            path=self.config.get("SETTINGS", "DB_PATH")
        )
        self.message_collection = self.vectorstore_client.get_or_create_collection(
            name="chat_messages", embedding_function=self.embed_fct
        )

    def initialize_db(self):
        """
        This method reads all of the dummy chat data and loads the data in the database
        if an entry does not exist yet for the entry. Use this function to avoid that
        messages get embedded again, even if a respective entry already exists in the
        vector database.
        """
        chat_messages = []
        chat_message_metadata = []

        for chat_data in self.dummy_data_access.read_chat_data():
            for message in chat_data["messages"]:
                chat_messages.append(message["content"])
                chat_message_metadata.append(
                    {
                        "user": message["senderUserId"],
                        "chat_id": chat_data["chatId"],
                        "timestamp": message["timestamp"],
                    }
                )

        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc)) for doc in chat_messages]
        unique_ids = list(set(ids))

        db_duplicates = self.message_collection.get(ids=unique_ids, include=[])

        seen_ids = set()

        ids_to_add = []
        docs_to_add = []
        metadata_to_add = []

        for doc, metadata, id in zip(chat_messages, chat_message_metadata, ids):
            if (
                id not in seen_ids
                and (seen_ids.add(id) or True)
                and id not in db_duplicates["ids"]
            ):
                ids_to_add.append(id)
                docs_to_add.append(doc)
                metadata_to_add.append(metadata)

        if len(ids_to_add) > 0:
            self.message_collection.add(
                ids=ids_to_add, documents=docs_to_add, metadatas=metadata_to_add
            )

    def get_langchain_retriever(self, chat_user, active_chat_id):
        langchain_chroma = Chroma(
            client=self.vectorstore_client,
            collection_name="chat_messages",
            embedding_function=OpenAIEmbeddings(openai_api_key=self.openai_key),
        )

        filter = {
            "$and": [
                {"user": {"$eq": chat_user}},
                {"chat_id": {"$ne": int(active_chat_id)}},
            ]
        }

        return langchain_chroma.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": float(
                    self.config.get("SETTINGS", "SIMILARITY_THRESHOLD")
                ),
                "k": int(self.config.get("SETTINGS", "TOP_K_SIM_MESSAGES_RETURNED")),
                "filter": filter,
            },
        )
