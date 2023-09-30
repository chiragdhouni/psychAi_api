
from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from langchain import ConversationChain

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat")
async def chat(query: str):
    import os
    os.environ["OPENAI_API_KEY"] = "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"
    import pinecone
    import nest_asyncio
    nest_asyncio.apply()
    from langchain.document_loaders.sitemap import SitemapLoader
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.prompts import ChatPromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain.llms import OpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.vectorstores import Pinecone
    from langchain.chains import ConversationChain
    pinecone.init(
        api_key="867228d2-f0a9-487c-bd53-4209240281da",  # find at app.pinecone.io
        environment="gcp-starter"  # next to api key in console
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    from langchain.chains import RetrievalQA
    llm = OpenAI(model_name='gpt-3.5-turbo')
    embeddings = OpenAIEmbeddings()
    index_name = "psync"
    prompt = "You are a mental health diagnosis expert, you will ask the user for inputs about themselves, and analyze the text they write to you. Then, you will ask a set of questions as per the requirement to assess the mental state/signs of problems faced by the user. After analysis, you will produce a diagnosis which will contain the most probable mental health issues, and produce the result in a user-sensitive way, do not be blunt and direct. If the user seems to be facing some normal/mild, reassure them and give them advices on improving their state."
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    messages = [
        SystemMessagePromptTemplate.from_template(prompt),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever(),
                                               memory=memory,
                                               chain_type="stuff",

                                               condense_question_prompt=qa_prompt)

    response =qa({"question": query}) # call LLM chatbot here
    return {"response": response['answer']}

#
#
# @app.get("/generate_response")
# async def generate_response(your_param: str = ""):
#     if str == "":
#         return {"message": "No query is provided"}
#     response = qa({"question": your_param})
#     return {"response": response['answer']}


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app,port=int(os.environ.get('PORT', 8080)), host="127.0.0.1")



