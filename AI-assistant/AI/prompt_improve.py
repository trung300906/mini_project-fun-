from google import genai

loopcount = 0
def improve_prompt(prompt: str) -> str:
    #using AI model to improve the prompt, 2 times loop for better results
    client = genai.Client(api_key="AIzaSyDCMgfjXlrdCVKMzFTyIL2-vpiAxw3ougo")
    global loopcount
    if loopcount < 2:
        loopcount += 1
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"You are a prompt optimizer. Rewrite the following prompt to make it clearer, more specific, and more effective for a language model to understand and respond to. Only return the improved version of the prompt, without any explanation or comments.Original Prompt:{prompt}"
        )
        improved_prompt = response.text.strip()
        return improve_prompt(improved_prompt)
    else:
        return prompt.strip()

