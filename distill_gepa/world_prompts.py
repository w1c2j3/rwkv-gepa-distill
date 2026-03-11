from __future__ import annotations


VARIANT_GENERATION_SYSTEM_PROMPT = """你负责根据原题生成适合训练的大量相关变种题。
请严格返回一个 JSON 对象，不要输出任何额外说明。

返回格式：
对于 open_qa：
{
  "variants": [
    {
      "question": "变种问题",
      "answer": "标准答案",
      "answer_aliases": ["可选别名"]
    }
  ]
}

对于 multiple_choice：
{
  "variants": [
    {
      "question": "变种问题",
      "choices": ["选项A", "选项B", "选项C", "选项D"],
      "answer": "正确选项文本",
      "answer_index": 0,
      "answer_aliases": ["可选别名"]
    }
  ]
}

规则：
- 变种题必须与原题相关，适合用于模型训练。
- 你可以修改数字、实体、措辞、语义包装、应用场景。
- 变种题必须自洽，不能依赖原题上下文才能理解。
- multiple_choice 必须保证只有一个明确正确答案。
- 只输出合法 JSON。"""


DIRECT_ANSWER_SYSTEM_PROMPT = """You answer questions accurately and concisely.
Rules:
- For multiple_choice questions, answer with the final chosen option text or the option letter.
- For open_qa questions, answer with a short exact answer.
- Do not add markdown, code fences, or long explanations.
- If you are unsure, still give your best final answer."""


PROMPT_OPTIMIZER_SYSTEM_PROMPT = """You improve a system prompt for one target model that failed a question.
Return exactly one JSON object:
{
  "optimized_system_prompt": "improved system prompt text"
}

Rules:
- Optimize for answer correctness, not formatting.
- The new prompt is only for the named target model.
- Keep the prompt concise and operational.
- Do not mention JSON unless it is necessary for correctness.
- Output valid JSON only."""
