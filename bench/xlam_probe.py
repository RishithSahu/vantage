"""Step 0 for a native function-calling specialist: load the checkpoint through the project's llama.cpp runtime, print its own chat template, reproduce the publisher's tool-call example, and report tokens / timings. Nothing project-specific is used.
    python tools_pin.py 8,9 bench/xlam_probe.py"""
import json, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from llama_cpp import Llama

PATH = os.path.join(HERE, "models", "xLAM-2-1B-fc-r-Q4_K_M.gguf")
t = time.time(); llm = Llama(model_path=PATH, n_ctx=4096, n_threads=2, n_batch=64, verbose=False); print(f"load {time.time() - t:.1f}s  n_vocab={llm.n_vocab()}")
tmpl = llm.metadata.get("tokenizer.chat_template", ""); print("chat template (first 1500 chars):\n", tmpl[:1500]); print("meta:", {k: v for k, v in llm.metadata.items() if k.startswith(("general.architecture", "general.name", "general.file_type", "llama.context", "tokenizer.ggml.bos", "tokenizer.ggml.eos"))})
tools = [{"type": "function", "function": {"name": "UserDetail", "parameters": {"type": "object", "title": "UserDetail", "properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}}, "required": ["name", "age"]}}}]
msgs = [{"role": "system", "content": "You are a helpful assistant that can use tools. You are developed by Salesforce xLAM team."}, {"role": "user", "content": "Extract Jason is 25 years old"}]
t = time.time(); out = llm.create_chat_completion(messages=msgs, tools=tools, tool_choice={"type": "function", "function": {"name": "UserDetail"}}, temperature=0.0, max_tokens=80); print("README example ->", json.dumps(out["choices"][0]["message"])[:400], f"({time.time() - t:.1f}s)", out["usage"])
t = time.time(); out = llm.create_chat_completion(messages=msgs, tools=tools, temperature=0.0, max_tokens=80); print("no tool_choice ->", json.dumps(out["choices"][0]["message"])[:400], f"({time.time() - t:.1f}s)", out["usage"])
# the rendered prompt, as the model receives it
try:
    from jinja2 import Template
    print("rendered:\n", Template(tmpl).render(messages=msgs, tools=tools, add_generation_prompt=True, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")[:2500])
except Exception as e: print("render failed", type(e).__name__, e)
