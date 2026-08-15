from langchain_core.messages import AIMessage, HumanMessage
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.core.config import logger
from app.models.user_model import User
from app.schemas.chat_schema import ChatBody
from app.services.rag_service import RAGService


router = APIRouter()

rag_service = RAGService()

chat_history: list = []


@router.post("/chat")
async def chat_action(
    request: ChatBody,
    current_user: User = Depends(get_current_user),
):
    global chat_history

    logger.info(
        f"User {current_user.id} sent a message: "
        f"{request.message}"
    )

    user_message = HumanMessage(
        content=request.message
    )

    answer = rag_service.ask(
        message=request.message,
        chat_history=chat_history,
    )

    chat_history.extend(
        [
            user_message,
            AIMessage(content=answer),
        ]
    )

    return {
        "data": answer,
    }