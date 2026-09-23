from dotenv import load_dotenv

from agent.agent import graph

load_dotenv()


response = graph.invoke(
    {
        "messages": [
            (
                "user",
                "What are the latest developments in artificial intelligence?"
            )
        ]
    }
)


for message in response["messages"]:
    print(message.content)  