from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough






#index



# search


#answer
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="Qwen/Qwen3-0.6B",
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es un assistant factuel.
Réponds uniquement depuis le contexte. Cite les sources.

Contexte : {contexte}""",
        ),
        ("human", "{question}"),
    ]
)


def formater_docs(docs: list[Document]):
    return "\n\n---\n\n".join(
        [
            f"[{doc.metadata.get('file_path', '?')}]\n{doc.page_content}"
            for doc in docs
        ]
    )


chain_hybrid = (
    {"contexte": hybrid | formater_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# res = bm25_retriever.invoke(
#     "What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM?"
# )

res = chain_hybrid.invoke(
    "What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM?"
)

print(res)
