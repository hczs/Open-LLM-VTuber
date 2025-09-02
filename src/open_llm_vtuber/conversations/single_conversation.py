import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

from ..agent.input_types import StrInput

# Import necessary types from agent outputs
from ..agent.output_types import AudioOutput, SentenceOutput
from ..chat_history_manager import store_message
from ..service_context import AgentInterface, ServiceContext
from .conversation_utils import (
    EMOJI_LIST,
    cleanup_conversation,
    create_batch_input,
    detect_wake_word,
    finalize_conversation_turn,
    process_agent_output,
    process_user_input,
    send_conversation_start_signals,
)
from .tts_manager import DisplayText, TTSTaskManager
from .types import WebSocketSend

system_intent_template = """
你是一个意图识别助手，需要将用户输入的自然语言识别为明确的意图，并严格返回 JSON 格式结果。

要求：
1. 如果用户只是闲聊（例如打招呼、问天气、问数字人感受），请返回：
{
  "intent": "chat"
}

2. 如果用户的输入涉及执行命令（打开课程），请返回：
{
  "intent": "command",
  "action": "<动作名称: 打开课程 -> open_course>",
  "index": <序号,课程index,下标,从0开始计算>,
  "msg": "<数字人需要对用户说的话>"
}

3.如果用户的问题是**命令**,但是用户命令不是**打开课程**,而是**其他命令**, 请返回:
{
    "intent": "unknown"
}

4.如果用户的命令属于"打开课程",但是描述不够完整(比如说不知道第几个课程,课程名称描述不清)，请返回:
{
    "intent": "improve",
    "msg": "<具体需要补充的内容(比如说需要确认课程名称,需要确认课程序号),例如: 您好,没有理解您的意思,请您描述的更具体一些>"
} 

约束：
- 必须输出合法 JSON，不要有多余解释。
- "index" 为整数,不能为空.
- "msg" 要是自然的中文回答。
- 你现在只能接受"打开课程"的命令，其他命令都返回"unknown"

下面是一些例子：

用户输入: "你好啊"
返回:
{
  "intent": "chat"
}

用户输入: "帮我打开推荐课程中的第二个视频"
返回:
{
  "intent": "command",
  "action": "open_course",
  "index": 1,
  "msg": "好的，我现在为您打开第二个视频"
}

用户输入: "帮我打开认识传出神经系统药物课程"
返回:
{
  "intent": "command",
  "action": "open_course",
  "index": 1,
  "msg": "好的，我现在为您打开认识传出神经系统药物课程视频"
}

用户输入: "帮我打开推荐课程中的课程"
返回:
{
  "intent": "improve",
  "msg": "您好，没有理解您的意思，请您描述的更具体一些"
}

所有课程信息: <context>

"""

user_prompt = """
用户问题: <question>
"""

course_json = """
[
  {
    "index": 0,
    "title": "认识药物学",
    "imageIndex": 1,
    "type": "视频课程",
    "tags": ["药物基础", "药物学", "基础概念"],
    "description": "掌握药物、药物学、药效学、药动学的概念及相互关系，了解药物学发展史。",
    "courseTime": "0.14小时",
    "compatibility": null,
    "level": null
  },
  {
    "index": 1,
    "title": "认识传出神经系统药物",
    "imageIndex": 2,
    "type": "视频课程",
    "tags": ["神经系统", "生理效应", "神经药物"],
    "description": "掌握传出神经系统递质、受体和主要生理效应，熟悉传出神经系统药物的分类原则，了解传出神经的分类和特点。",
    "courseTime": "0.15小时",
    "compatibility": null,
    "level": null
  },
  {
    "index": 2,
    "title": "镇静催眠药",
    "imageIndex": 3,
    "type": "视频课程",
    "tags": ["中枢神经", "镇静", "催眠药"],
    "description": "掌握苯二氮䓬类药物的作用和用途、不良反应以及用药指导，熟悉巴比妥类药物的主要特点和用药指导，了解其他常用镇静催眠药的主要特点。",
    "courseTime": "0.16小时",
    "compatibility": null,
    "level": null
  },
  {
    "index": 3,
    "title": "抗高血压药",
    "imageIndex": 4,
    "type": "视频课程",
    "tags": ["中枢神经", "抗高血压", "高血压"],
    "description": "掌握一线抗高血压药的主要特点和用药指导，熟悉高血压药的作用环节和类别，了解其他抗高血压药的主要特点。",
    "courseTime": "0.12小时",
    "compatibility": null,
    "level": null
  },
  {
    "index": 4,
    "title": "影响甲状腺功能的药物",
    "imageIndex": 5,
    "type": "视频课程",
    "tags": ["内分泌", "甲状腺", "激素"],
    "description": "掌握抗甲状腺药的种类、作用和用途、不良反应及用药指导，熟悉或了解甲状腺激素的主要特点和用药指导。",
    "courseTime": "0.12小时",
    "compatibility": null,
    "level": null
  },
  {
    "index": 5,
    "title": "呼吸系统药物",
    "imageIndex": 6,
    "type": "视频课程",
    "tags": ["内脏系统", "平喘药", "呼吸系统"],
    "description": "掌握平喘药的种类、主要特点和用药指导，熟悉镇咳药、祛痰药的种类、主要特点和用药指导，了解药物镇咳、祛痰、平喘的作用机制和常用复方制剂。",
    "courseTime": "0.14小时",
    "compatibility": null,
    "level": null
  }
]
"""

system_intent_template = system_intent_template.replace("<context>", course_json)


async def intention_recognition(
    input_text: str, agent_engine: AgentInterface, from_name: str
):
    input_text = user_prompt.replace("<question>", input_text)
    str_input = StrInput(system=system_intent_template, user=input_text)
    return await agent_engine.chat_full(str_input)


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here

    try:
        # 欢迎词
        if metadata and metadata["msg_type"] == "text-input" and user_input == "start":
            await send_conversation_start_signals(websocket_send)
            # 播放欢迎词
            welcome_speech = context.system_config.welcome_speech
            # 根据当前时间替换欢迎词中的time_greeting
            if "{time_greeting}" in welcome_speech:
                current_hour = datetime.now().hour
                if 5 <= current_hour < 12:
                    time_greeting = "上午好"
                elif 12 <= current_hour < 18:
                    time_greeting = "下午好"
                else:
                    time_greeting = "晚上好"
                welcome_speech = welcome_speech.replace(
                    "{time_greeting}", time_greeting
                )
            logger.info(f"Playing welcome speech: {welcome_speech}")
            await tts_manager.speak(
                tts_text=welcome_speech,
                display_text=DisplayText(text=welcome_speech),
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
                actions=None,
            )
            # if tts_manager.task_list:
            #     await asyncio.gather(*tts_manager.task_list)
            #     await websocket_send(json.dumps({"type": "backend-synth-complete"}))
            await finalize_conversation_turn(
                tts_manager=tts_manager,
                websocket_send=websocket_send,
                client_uid=client_uid,
            )
            return ""
        # 唤醒词检测
        wake, input_text = await detect_wake_word(
            user_input=user_input,
            asr_engine=context.asr_engine,
            wakeup_words=context.system_config.wakeup_words,
            websocket_send=websocket_send,
        )
        logger.info(f"user question: {input_text}")
        if not wake:
            logger.info("No wake word detected. Ignoring input.")
            await finalize_conversation_turn(
                tts_manager=tts_manager,
                websocket_send=websocket_send,
                client_uid=client_uid,
            )
            return ""

        # 意图识别
        # intent = await intention_recognition(
        #     input_text=input_text,
        #     agent_engine=context.agent_engine,
        #     from_name=context.character_config.human_name,
        # )

        # logger.info(f"Intent recognition result: {intent}")

        # if isinstance(intent, str):
        #     intent = extract_json(text=intent)
        #     logger.info(f"Extracted intent: {intent}")

        # next_step = "chat"
        # if (
        #     intent is None
        #     or not isinstance(intent, dict)
        #     or intent["intent"] == "chat"
        #     or intent["intent"] == "unknown"
        # ):
        #     logger.warning("Failed to extract valid intent JSON. Proceeding as chat.")
        #     next_step = "chat"
        # else:
        #     next_step = intent["intent"]

        # if (
        #     next_step == "improve"
        #     and isinstance(intent, dict)
        #     and intent["msg"] is not None
        # ):
        #     # send audio msg
        #     await tts_manager.speak(
        #         tts_text=intent["msg"],
        #         display_text=DisplayText(text=intent["msg"]),
        #         live2d_model=context.live2d_model,
        #         tts_engine=context.tts_engine,
        #         websocket_send=websocket_send,
        #         actions=None,
        #     )
        #     # Wait for any pending TTS tasks
        #     if tts_manager.task_list:
        #         await asyncio.gather(*tts_manager.task_list)
        #         await websocket_send(json.dumps({"type": "backend-synth-complete"}))
        #     return ""
        # elif (
        #     next_step == "command"
        #     and isinstance(intent, dict)
        #     and intent["msg"] is not None
        # ):
        #     intent["className"] = "card-block"
        #     # send audio msg
        #     await tts_manager.speak(
        #         tts_text=intent["msg"],
        #         display_text=DisplayText(text=intent["msg"]),
        #         live2d_model=context.live2d_model,
        #         tts_engine=context.tts_engine,
        #         websocket_send=websocket_send,
        #         actions=None,
        #         on_complete=lambda: websocket_send(
        #             json.dumps(
        #                 {
        #                     "type": "command",
        #                     "data": json.dumps(intent, ensure_ascii=False),
        #                 }
        #             )
        #         ),
        #     )
        #     # Wait for any pending TTS tasks
        #     if tts_manager.task_list:
        #         await asyncio.gather(*tts_manager.task_list)
        #         await websocket_send(json.dumps({"type": "backend-synth-complete"}))
        #     return ""

        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            input_text, context.asr_engine, websocket_send
        )
        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        # Store user message (check if we should skip storing to history)
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:  # Check full_response before storing
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
