from google import genai
import database.init_data as init_data
import AI.prompt_improve as prompt_improve
client = genai.Client(api_key="AIzaSyDCMgfjXlrdCVKMzFTyIL2-vpiAxw3ougo")

def ask_model(text: str) -> str:
    if text.lower() == "clear history":
        init_data.clear_history()
        return "Conversation history cleared."
    # read history content
    history = init_data.load_history()
    if history:
        content = init_data.convert_history_to_content(history)
        text = f"{content}\n\nUser: {text}"
    improved_prompt = prompt_improve.improve_prompt(text)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=improved_prompt
    )
    init_data.save_to_history(text, response.text)
    return response.text
