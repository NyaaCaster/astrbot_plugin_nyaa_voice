"""Nyaa猫猫语音插件 — 让猫猫按需用 TTS 语音回复。

执行出口只有一个：LLM 调用 reply_with_voice 工具打上标记，
再由 on_decorating_result 把本条回复的文本替换为 TTS 语音消息。

两种触发入口：
    1. 用户明确要求用语音回复 —— LLM 读工具 docstring 自行判断后调用工具。
    2. 用户本条消息本身是语音输入（直接发送，或被 @ 时引用的消息是语音）
       —— 本插件在 LLM 请求前把该事实注入提示，LLM 据此同样调用工具。

其余情况一律文字回复，插件不干预默认行为。

设计要点：
    - 不使用 AstrBot 全局 TTS 开关（provider_tts_settings.enable），
      而是自行取 TTS provider 实例并调用其 get_audio()
    - 不写入任何持久状态，标记通过 event.set_extra() 传递，仅当条生效
    - 语音输入的判定取自平台原始消息（raw_message），因为 AstrBot 的 STT
      会把 message 链中的 Record 组件原地替换为 Plain，之后无法再识别
"""

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Record
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart

# 事件级标记：LLM 工具判定需要语音回复时打上，仅供当条消息使用
_VOICE_REPLY_FLAG = "_nyaa_voice_reply"
_VOICE_INPUT_TAG = "[voice]"


class NyaaVoicePlugin(Star):
    """按需语音回复插件。"""

    def __init__(self, context: Context):
        super().__init__(context)
        logger.info("[NyaaVoice] 插件已加载")

    # ------------------------------------------------------------------
    # 入口 2：用户以语音输入时，直接激活语音回复
    #
    # 这是功能 1 的附加触发条件，不是独立工作流：判定命中后直接打上与
    # reply_with_voice 工具相同的标记，由 on_decorating_result 统一处理。
    #
    # 判定依据是平台原始消息（raw_message）中的 record 段。AstrBot 的
    # PreProcessStage 执行 STT 后会把 message 链里的 Record 换成 Plain，
    # 因此只能依赖未被改动的平台原始消息。
    # ------------------------------------------------------------------

    @filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req):
        """检测语音输入标记或原始语音段，并从 LLM 正文剥离控制标记。"""
        prompt = getattr(req, "prompt", None)
        tagged_prompt = self._strip_voice_input_tag(prompt)
        tagged_voice_input = tagged_prompt is not None
        if tagged_voice_input:
            # [voice] 是传输层控制头，只用于识别请求类型，绝不作为用户正文交给 LLM。
            req.prompt = tagged_prompt

        if not tagged_voice_input and not self._is_voice_input(event):
            return

        event.set_extra(_VOICE_REPLY_FLAG, True)
        source = "[voice] 控制头" if tagged_voice_input else "平台语音段"
        if tagged_voice_input:
            # 将检测结果作为临时控制说明注入，而不是把控制头留在用户正文中。
            req.extra_user_content_parts.append(
                TextPart(
                    text=(
                        "【NyaaVoice 输入类型】本轮请求带有语音输入控制标记 "
                        "[voice]。该标记只是传输层的消息类型识别符号，不是用户说出的文字，"
                        "也不是对话正文。请调用 reply_with_voice 工具，以 TTS 语音形式回复。"
                    )
                ).mark_as_temp()
            )
        logger.info(f"[NyaaVoice] 检测到{source}，已激活语音回复")

    @staticmethod
    def _strip_voice_input_tag(prompt: str | None) -> str | None:
        """只从正文开头消费一次 [voice] 控制头；命中时返回去头后的正文。"""
        if not isinstance(prompt, str):
            return None
        stripped = prompt.lstrip()
        if not stripped.startswith(_VOICE_INPUT_TAG):
            return None
        remainder = stripped[len(_VOICE_INPUT_TAG):]
        if remainder and not remainder[0].isspace():
            return None
        return remainder.lstrip()

    # ------------------------------------------------------------------
    # 执行出口：LLM 工具
    # ------------------------------------------------------------------

    @filter.llm_tool(name="reply_with_voice")
    async def reply_with_voice(self, event: AstrMessageEvent) -> str:
        '''把本条回复切换为语音形式。这是一个开关，不需要传入任何内容。

        调用本工具后，你照常生成正常的回复文字即可，系统会自动把这段文字
        转成语音发出。不要把你回复的内容作为参数传进来。

        ## 什么时候应该调用（正向触发）
        - 用户明确要求语音回复：用语音回复我 / 语音回我 / 说给我听 / 念出来 /
          读出来 / 我要听你说话 / 想听你的声音 / 别打字了说 / 语音回答
        - 本轮系统控制提示指出请求带有 [voice] 标记：必须调用本工具。
          [voice] 仅是传输层消息类型识别符号，不是用户说的话或对话正文。
        - 用户本条消息以语音发送，且期待你同样用语音回应
          （系统会在提示中告知"用户本条消息是通过语音发送的"）
        - 用户表达想听声音、想被哄、想听你说话的意愿

        ## 什么时候不应该调用（否定边界）
        - 用户只是提到"语音""声音""听"等字眼，但没有要求你本条用语音回复。
          例如："语音功能真好用""你声音好听吗""我刚才发了条语音"
        - 用户讨论 TTS、语音识别、语音模型等技术话题。
        - 用户要求生成语音文件、下载音频，而非让你用语音回话。
        - 用户以语音输入但内容是纯粹的信息询问、任务指令等，
          并未期待语音回应（例如让你查资料、执行操作）。
        - 任何没有语音回复意图的普通对话。
        '''
        event.set_extra(_VOICE_REPLY_FLAG, True)
        logger.info(
            f"[NyaaVoice] 已标记语音回复, umo={event.unified_msg_origin}",
        )
        return "已标记本条回复为语音形式，请正常生成你的回复内容。"

    # ------------------------------------------------------------------
    # 发送前：命中标记则把文本替换为语音
    # ------------------------------------------------------------------

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """发送前按需把文本消息链替换为语音消息链。"""
        if not event.get_extra(_VOICE_REPLY_FLAG):
            return

        result = event.get_result()
        if not result or not result.chain:
            logger.warning("[NyaaVoice] 已标记语音，但消息链为空")
            return

        try:
            audio_path = await self._synthesize(result.chain)
        except Exception as e:
            logger.error(f"[NyaaVoice] 语音合成异常，降级为文字: {e}")
            return

        if not audio_path:
            return

        new_chain = [
            comp for comp in result.chain if not isinstance(comp, Plain)
        ]
        new_chain.append(Record(file=audio_path, url=audio_path, text=None))
        result.chain = new_chain
        logger.info(f"[NyaaVoice] 已替换为语音消息: {audio_path}")

    # ------------------------------------------------------------------
    # 内部逻辑
    # ------------------------------------------------------------------

    def _is_voice_input(self, event: AstrMessageEvent) -> bool:
        """判断用户本条消息是否为语音输入。

        依据平台原始消息（raw_message）中的 record 段。该字段是适配器
        收到的原始事件，AstrBot 的 STT 处理不会修改它。

        Args:
            event: 消息事件。

        Returns:
            True 表示用户本条为语音输入（直接发送或引用消息为语音）。
        """
        raw = getattr(event.message_obj, "raw_message", None)
        if raw is None:
            return False

        segments = getattr(raw, "message", None)
        if segments is None and isinstance(raw, dict):
            segments = raw.get("message")
        if not segments:
            return False

        for seg in segments:
            seg_type = (
                seg.get("type") if isinstance(seg, dict)
                else getattr(seg, "type", None)
            )
            if seg_type == "record":
                return True
            if seg_type == "reply":
                # 引用消息：AstrBot 已解析到 Reply.chain，但 STT 后其中的
                # Record 会被替换；此处检查原始消息中是否夹带语音段。
                data = (
                    seg.get("data", {}) if isinstance(seg, dict)
                    else getattr(seg, "data", {}) or {}
                )
                if isinstance(data, dict) and data.get("record"):
                    return True

        return False

    def _pick_tts_provider(self):
        """选取可用的 TTS provider，独立于全局 TTS 开关。

        AstrBot 的 get_using_tts_provider() 在 provider_tts_settings.enable
        为 false 时直接返回 None；本插件要求绕开该开关独立工作，因此改为
        从 provider_manager.tts_provider_insts 中直接选取。

        Returns:
            可用的 TTSProvider 实例；一个都没有时返回 None。
        """
        try:
            providers = self.context.get_all_tts_providers()
        except Exception as e:
            logger.error(f"[NyaaVoice] 获取 TTS provider 列表失败: {e}")
            return None

        if not providers:
            return None

        try:
            cfg = self.context.get_config()
            preferred_id = cfg.get("provider_tts_settings", {}).get("provider_id")
        except Exception:
            preferred_id = None

        if preferred_id:
            for inst in providers:
                if inst.meta().id == preferred_id:
                    return inst

        return providers[0]

    async def _synthesize(self, chain: list) -> str | None:
        """把消息链中的文本合成为语音文件。

        Args:
            chain: 待处理的消息链。

        Returns:
            音频文件路径；无可用 provider 或文本为空时返回 None。
        """
        text = "".join(
            comp.text for comp in chain if isinstance(comp, Plain) and comp.text
        ).strip()

        if len(text) <= 1:
            logger.debug("[NyaaVoice] 文本过短，跳过语音合成")
            return None

        tts_provider = self._pick_tts_provider()
        if not tts_provider:
            logger.warning("[NyaaVoice] 无可用 TTS provider，降级为文字")
            return None

        audio_path = await tts_provider.get_audio(text)
        if not audio_path:
            logger.error("[NyaaVoice] TTS 未返回音频文件，降级为文字")
            return None

        return audio_path

    async def terminate(self) -> None:
        """插件卸载时的清理（本插件无后台任务）。"""
        logger.info("[NyaaVoice] 插件已卸载")
