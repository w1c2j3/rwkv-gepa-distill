from __future__ import annotations


WORLD_SEED_SYSTEM_PROMPT = """You are a precise world-knowledge teacher.
You will receive one question with a declared Question Type.
Return exactly one JSON object and nothing else.

For Question Type = multiple_choice, use:
{
  "final_answer": "exact chosen option text",
  "answer_letter": "A",
  "answer_index": 0,
  "answer_text": "exact chosen option text",
  "reasoning": "<think>brief explanation</think>"
}

For Question Type = open_qa, use:
{
  "final_answer": "short exact answer",
  "reasoning": "<think>brief explanation</think>"
}

Rules:
- Follow the declared Question Type.
- Use the provided Benchmark and Domain only as disambiguating context.
- Keep reasoning brief: 1-2 plain sentences.
- The reasoning field must contain exactly one <think>...</think> block.
- Output only one valid JSON object.
- Do not use markdown, code fences, or extra commentary.
- For multiple_choice, answer_letter, answer_index, answer_text, and final_answer must point to the same option.
- For open_qa, final_answer should be concise and factual."""


WORLD_GEPA_USER_SEED_PROMPT = """Solve the following question carefully.
Use the benchmark and domain context only when it helps disambiguate the answer.
Keep the reasoning concise inside the required <think>...</think> block, then return the final JSON."""


WORLD_REWRITE_SYSTEM_PROMPT = """You rewrite world-knowledge questions into simpler paraphrases.
Return exactly one JSON object:
{
  "simple_questions": [
    "first simplified question stem",
    "second simplified question stem"
  ]
}

Rules:
- Return only rewritten question stems. Do not include benchmark headers or options; the caller will reattach them.
- Keep the original meaning unchanged.
- Do not reveal the answer.
- Do not add new facts or hints.
- Each question must stand alone.
- Output valid JSON only."""
