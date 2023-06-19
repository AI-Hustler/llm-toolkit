import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")



TEXT_EMBEDDING_CHUNK_SIZE=200
MAX_TEXTS_TO_EMBED_BATCH_SIZE=100


# Create embeddings for a text using a tokenizer and an OpenAI engine
def create_embeddings_for_text(model, text):
    
    """Return a list of tuples (text_chunk, embedding) and an average embedding for a text."""
    token_chunks = list(chunks(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

    # Split text_chunks into shorter arrays of max length 10
    text_chunks_arrays = [text_chunks[i:i+MAX_TEXTS_TO_EMBED_BATCH_SIZE] for i in range(0, len(text_chunks), MAX_TEXTS_TO_EMBED_BATCH_SIZE)]


    # Call get_embeddings for each shorter array and combine the results
    embeddings = []
    for text_chunk in text_chunks_arrays[0]:
        # print(text_chunks_array)
        embeddings_response = model.embed_query([text_chunk])
        embeddings.append(embeddings_response)
        
    text_plus_embeddings = list(zip(text_chunks, embeddings))

    return text_plus_embeddings

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j