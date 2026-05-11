"""Smoke test for distill_demo.ipynb: 1 sample, 2 epochs, just confirm the loop runs."""
import gc, json, torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).parent
TEACHER_ID, STUDENT_ID = 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-0.8B'
FPS, TOPK, TEMPERATURE, KL_WEIGHT = 2.0, 32, 2.0, 0.7
SYSTEM_PROMPT = ('You answer questions about short sports videos. This is a classification task: '
                 'every answer is a single label from a fixed vocabulary. '
                 'Output ONLY the answer label — 1 to 3 words, no sentences, no punctuation.')


def resolve(video_key):
    sport, stem = video_key.split('/', 1)
    v = ROOT / 'videos'
    exts = ('.avi', '.mp4', '.mkv', '.webm')
    cands = []
    if sport == 'fg':
        for d in (v / f'finegym_0{i}' for i in range(1, 6)):
            cands += [d / f'{stem}{e}' for e in exts]
    elif sport == 'volleyball':
        cands += [v / 'volleyball' / 'volleyball' / f'{stem}{e}' for e in exts]
    else:
        cands += [v / sport / f'{stem}{e}' for e in exts]
    return next((p for p in cands if p.exists()), None)


def build_inputs(processor, vp, q, gold=None):
    msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
            {'role': 'user', 'content': [{'type': 'video', 'video': str(vp), 'fps': FPS},
                                          {'type': 'text', 'text': q}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    # Run the full processor on the prompt alone to get the true prompt_len (video tokens expanded)
    prompt_only = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors='pt')
    prompt_len = prompt_only['input_ids'].shape[1]
    full = text + gold if gold else text
    inp = processor(text=[full], images=ii, videos=vi, padding=True, return_tensors='pt').to('cuda')
    return inp, prompt_len


train = json.loads((ROOT / 'meta-data' / 'train.json').read_text())
sample = None
for r in train:
    if resolve(r['video']) is not None:
        sample = (r, resolve(r['video'])); break
r, vp = sample
print(f"sample: qa_id={r['qa_id']}  Q={r['question']!r}  gold={r['answer']!r}")

# tokenizer parity
t_tok = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
s_tok = AutoTokenizer.from_pretrained(STUDENT_ID, trust_remote_code=True)
assert t_tok.get_vocab() == s_tok.get_vocab(), 'vocab mismatch'
print(f'✓ tokenizers match, vocab={len(t_tok.get_vocab())}')

# phase 1 — teacher
proc_t = AutoProcessor.from_pretrained(TEACHER_ID, trust_remote_code=True)
teacher = AutoModelForImageTextToText.from_pretrained(
    TEACHER_ID, dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True).eval()
print(f'teacher loaded: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

inputs, plen = build_inputs(proc_t, vp, r['question'], gold=r['answer'])
glen = inputs['input_ids'].shape[1] - plen
gold_ids = inputs['input_ids'][0, plen:plen+glen].clone()
with torch.inference_mode():
    logits = teacher(**inputs).logits[0, plen-1:plen-1+glen].float()
tk_vals, tk_idx = logits.topk(k=TOPK, dim=-1)
in_tk = (tk_idx == gold_ids.unsqueeze(-1)).any(-1)
if not in_tk.all():
    m = (~in_tk).nonzero(as_tuple=True)[0]
    lo = tk_vals[m].argmin(-1)
    tk_idx[m, lo] = gold_ids[m]; tk_vals[m, lo] = logits[m, gold_ids[m]]
print(f'teacher extraction: gold_len={glen}, topK shape={tuple(tk_vals.shape)}')

teacher_cache = {'gold_ids': gold_ids.cpu(), 'tk_vals': tk_vals.cpu(), 'tk_idx': tk_idx.cpu(),
                 'row': r, 'vp': vp}
del teacher, proc_t; gc.collect(); torch.cuda.empty_cache()
print(f'after teacher unload: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

# phase 2 — student
proc_s = AutoProcessor.from_pretrained(STUDENT_ID, trust_remote_code=True)
student = AutoModelForImageTextToText.from_pretrained(
    STUDENT_ID, dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
student.train(); student.gradient_checkpointing_enable(); student.config.use_cache = False
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)

for epoch in range(2):
    inp_s, plen_s = build_inputs(proc_s, teacher_cache['vp'], teacher_cache['row']['question'],
                                  gold=teacher_cache['row']['answer'])
    glen_s = inp_s['input_ids'].shape[1] - plen_s
    goldid_s = inp_s['input_ids'][0, plen_s:plen_s+glen_s]
    slog = student(**inp_s).logits[0, plen_s-1:plen_s-1+glen_s].float()

    t_vals = teacher_cache['tk_vals'].to('cuda')
    t_idx = teacher_cache['tk_idx'].to('cuda').long()
    s_on_topk = slog.gather(-1, t_idx)
    p_t = (t_vals / TEMPERATURE).softmax(-1)
    logq_s = (s_on_topk / TEMPERATURE).log_softmax(-1)
    kl = F.kl_div(logq_s, p_t, reduction='batchmean') * (TEMPERATURE ** 2)
    ce = F.cross_entropy(slog, goldid_s)
    loss = KL_WEIGHT * kl + (1 - KL_WEIGHT) * ce

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    print(f'epoch {epoch+1}: CE={ce.item():.3f}  KL={kl.item():.3f}  loss={loss.item():.3f}')

print(f'\npeak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
print('SMOKE TEST OK')
