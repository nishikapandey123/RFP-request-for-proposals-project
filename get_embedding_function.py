


# from langchain_openai import OpenAIEmbeddings

# def get_embedding_function():
#     return OpenAIEmbeddings()


from langchain.embeddings.openai import OpenAIEmbeddings  # Ensure you are importing the correct class

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify the model if needed

