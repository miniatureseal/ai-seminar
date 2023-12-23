import configparser
import chromadb

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pyprojroot.here import here

from openai_key_retrieval import get_openai_key
from db_helper import ChromaDBHelper


"""
This script is used to test the similarity search functionality of ChromaDB.
You can use it to query all the data which was read in from the dummy data
and find similar chunks in the database. You only need to change the first two
variables to your liking.
"""

user_to_query = "etaylor"
query = "We prioritize building long-term relationships, by providing value to our clients and ensuring that we are meeting their needs. Can you share your experience in client relationship management? Any specific examples that showcase your strengths in this area?"


db_helper = ChromaDBHelper()
db_helper.initialize_db()

config = configparser.ConfigParser()
config.read(here("config.ini"))
chromadb_client = chromadb.PersistentClient(path=config.get("SETTINGS", "DB_PATH"))

langchain_chroma = Chroma(
    client=chromadb_client,
    collection_name="chat_messages",
    embedding_function=OpenAIEmbeddings(api_key=get_openai_key()),
)

filter = {"user": {"$eq": user_to_query}}

results = langchain_chroma.similarity_search_with_relevance_scores(
    query=query,
    k=int(config.get("SETTINGS", "TOP_K_SIM_MESSAGES_RETURNED")),
    score_threshold=float(config.get("SETTINGS", "SIMILARITY_THRESHOLD")),
    filter=filter,
)

print(results)
