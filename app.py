import chainlit as cl
from query_processor import QueryProcessor  # Your existing QueryProcessor class
import asyncio
import os

# Initialize the query processor
processor = None

@cl.on_chat_start
async def start():
    global processor
    processor = QueryProcessor()
    
    await cl.Message(
        content="Hi! I can help you with the following queries:\n"
                "1. Get document summaries\n"
                "2. Find entities/POS in documents\n"
                "3. Find documents containing specific entities\n"
                "4. Find documents about specific topics\n"
                "5. Find documents with similar images"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    
    files = None
    if message.elements:
        files = message.elements

    try:
        intent, _ = processor.classify_intent(query)

        if intent == "SIMILAR_IMAGE_DOCS":
            if files:
                file_path = os.path.relpath(files[0].path)
                result = await processor.process_query(query, file_path)
            else:
                await cl.Message(
                    content="Please provide an image for image similarity search.\n"
                            "You can upload an image using the file upload button."
                ).send()
                return
        else:
            result = await processor.process_query(query)

        # Create a message element and explicitly set the response format
        msg = cl.Message(content=str(result))
        
        # Disable any automatic code formatting
        msg.markdown = False
        
        # Send a single response
        await msg.send()

    except Exception as e:
        await cl.Message(
            content=f"An error occurred: {str(e)}\n"
                    "Please try rephrasing your query."
        ).send()

@cl.on_settings_update
async def setup_settings(settings):
    print("Settings updated:", settings)