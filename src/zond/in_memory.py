from ast import While

from dotenv import load_dotenv
from langchain_classic.indexes import vectorstore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load documents
DATA_PATH = "../../data/books"
loader = DirectoryLoader(DATA_PATH, glob="*.md")
documents = loader.load()

#loader = TextLoader("data.txt", encoding="utf-8")
#documents = loader.load()

# Split doc in chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# create embeddings
embeddings = OpenAIEmbeddings()

# store document chunks in  FAISS vector db
vectorstore = FAISS.from_documents(docs, embeddings)

# create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# create llm
llm = ChatOpenAI(model="gpt-5.4", temperature=0)

# create prompt template
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
 Use only the provided context to answer the question.
 If the answer is not in the context, say: "I could not find in the document" 
 Context:
 {context}
 Question:
 {question}
 """
                                          )
# Start question answer loop
print("RAG App is ready. Type 'Exit' to stop. \n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        print("Thank you for your time. See you next time!")
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # build chain
    chain = prompt | llm | StrOutputParser()

    # get response
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print ("\n Answer: ")
    print(response)
    print("\n" + "-"*20 + "\n")