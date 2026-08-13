import pyttsx3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. INITIALIZE THE ROBOT'S VOICE (Text-to-Speech)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) 
engine.setProperty('rate', 170)

def robot_speak(text):
    print(f"\nRobot: {text}")
    engine.say(text)
    engine.runAndWait()

# 2. INITIALIZE THE ROBOT'S BRAIN (AI Chatbot Model)
print("Loading robot consciousness... (Bypassing microphone requirements)")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
print("Robot consciousness loaded successfully!")

# Keep track of conversation history
chat_history_ids = None

robot_speak("Hello. My systems are online. Type something in the terminal to talk to me.")

# 3. THE TEXT-INPUT LOOP
while True:
    try:
        # Instead of listening to a mic, it safely waits for your typed text
        user_text = input("\nYou: ")
        
        # Check for exit commands
        if "shutdown" in user_text.lower() or "goodbye" in user_text.lower():
            robot_speak("Deactivating system. Goodbye.")
            break
            
        if not user_text.strip():
            continue

        # --- THINKING PHASE (AI GENERATION) ---
        new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,       
            do_sample=True,               
            top_k=50, 
            top_p=0.95
        )

        robot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:], skip_special_tokens=True)
        
        # --- SPEAKING PHASE ---
        robot_speak(robot_response)

    except KeyboardInterrupt:
        print("\nSystem manually interrupted. Exiting.")
        break
