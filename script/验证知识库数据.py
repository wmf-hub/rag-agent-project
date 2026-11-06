from app.rag.vector_store import get_milvus_client, COLLECTION_NAME

c=get_milvus_client()
print(c.get_load_state(COLLECTION_NAME), c.get_collection_stats(COLLECTION_NAME))