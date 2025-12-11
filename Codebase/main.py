import os, time, torch, soundfile as sf, winsound
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from kokoro import KPipeline
from sentence_transformers import SentenceTransformer, util


#here we are calling laod_dotenv to laod the env variabeles
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN missing in .env file.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

print("🔍 Loading semantic evaluation model…")
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

def is_similar(q1, q2, threshold=0.70):
    emb1 = semantic_model.encode(q1, convert_to_tensor=True)
    emb2 = semantic_model.encode(q2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item() >= threshold


print("🎤 Loading Interviewer Model (Gemma + LoRA)…")

base = "google/gemma-3-1b-it"
adapter = "Shlok307/ai_interview-lora"

interviewer_tokenizer = AutoTokenizer.from_pretrained(base, token=HF_TOKEN)

interviewer_model = AutoModelForCausalLM.from_pretrained(
    base,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

interviewer_model = PeftModel.from_pretrained(interviewer_model, adapter, token=HF_TOKEN)
interviewer_model.eval()


print("🟣 Loading Evaluation Model (LLaMA 1B)…")

eval_name = "Shlok307/llama-1b-4bit"
eval_tokenizer = AutoTokenizer.from_pretrained(eval_name, token=HF_TOKEN)

eval_model = AutoModelForCausalLM.from_pretrained(
    eval_name,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
eval_model.eval()


tts = KPipeline(lang_code="a")

def speak(text):
    audio_out = tts(text, voice="am_liam", speed=0.85, split_pattern=r"\n+")
    for _, _, audio in audio_out:
        sf.write("interviewer.wav", audio, 24000)
        winsound.PlaySound("interviewer.wav", winsound.SND_FILENAME)
        break
    time.sleep(0.1)


def generate_ground_truth(question):
    prompt = f"Give a short and correct one-line answer. Question: {question}\nAnswer:"
    inputs = eval_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = eval_model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.2,
            eos_token_id=eval_tokenizer.eos_token_id
        )

    text = eval_tokenizer.decode(output[0], skip_special_tokens=True)
    ans = text.replace(prompt, "").strip()
    return ans.split("\n")[0].strip()


def semantic_score(candidate, ideal):
    emb1 = semantic_model.encode(candidate, convert_to_tensor=True)
    emb2 = semantic_model.encode(ideal, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()

    if score >= 0.70: return 100, "Fully correct."
    if score >= 0.45: return 70, "Partially correct."
    if score >= 0.25: return 40, "Some relevance."
    return 0, "Incorrect."

# if want to change the subject or domain then replace the word web dev with the requried subject
def generate_question(answer, previous_questions):
    prompt = f"""
You are a strict web development interviewer.
Ask EXACTLY ONE short technical question related to the candidate’s last answer.
Rules:
- Do NOT repeat or rephrase previous questions.
- Do NOT number the question.
- ONE sentence only.

Candidate answer: "{answer}"

Interviewer question:
"""

    for attempt in range(5):
        inputs = interviewer_tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = interviewer_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.6,
                do_sample=True,
                eos_token_id=interviewer_tokenizer.eos_token_id,
            )

        full = interviewer_tokenizer.decode(output[0], skip_special_tokens=True)
        q = full[len(prompt):].strip().split("\n")[0].replace('"', "").strip()

        if not q or q.endswith(":"): 
            continue
        if q in previous_questions:
            continue
        if any(is_similar(q, old) for old in previous_questions):
            continue

        return q

    return "What challenge have you faced recently in web development?"

# main function
def main():
    print("\n🔵 Interview System Ready. Type 'exit' to quit.\n")

    question = "So, shall we start the interview?"
    print("Interviewer:", question)
    speak(question)

    last_q = None
    previous_questions = []

    while True:
        candidate = input("\n🟢 Candidate: ")

        if candidate.lower() == "exit":
            break

        if last_q:
            ideal = generate_ground_truth(last_q)
            score, explanation = semantic_score(candidate, ideal)

            print(f"\n✅ Score: {score}/100 — {explanation}")
            print(f"💡 Ideal answer: {ideal}\n")

        next_q = generate_question(candidate, previous_questions)
        previous_questions.append(next_q)

        print("Interviewer:", next_q)
        speak(next_q)

        last_q = next_q


if __name__ == "__main__":
    main()
