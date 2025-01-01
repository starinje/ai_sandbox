from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from dotenv import load_dotenv
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory,
)

from langchain_community.chat_message_histories import FileChatMessageHistory

load_dotenv()

chat = ChatOpenAI(verbose=True)


memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"),
    llm=chat,
    memory_key="messages",
    return_messages=True,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">> ")

    print(f"You entered: {content}")

    result = chain({"content": content})

    print(result["text"])
