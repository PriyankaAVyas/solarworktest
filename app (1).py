import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import gradio as gr

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# Initialize Groq model
model = Groq(id="llama3-70b-8192")

# Agent 1: Homework Helper
homework_helper_agent = Agent(
    name="HomeworkHelperAgent",
    model=model,
    tools=[DuckDuckGo()],
    instructions=["Give clear and simple academic explanations. Always include sources if searched."],
    show_tool_calls=True,
    markdown=True,
)

# Agent 2: Study Tips Coach
study_tips_coach_agent = Agent(
    name="StudyTipsCoachAgent",
    model=model,
    instructions=["Give encouraging and practical study tips. Include examples where possible."],
    markdown=True,
)

# Routing function
def study_buddy_ai(user_prompt):
    academic_keywords = ["explain", "what is", "define", "solve", "difference", "example", "describe"]
    if any(keyword in user_prompt.lower() for keyword in academic_keywords):
        response = homework_helper_agent.run(user_prompt)
    else:
        response = study_tips_coach_agent.run(user_prompt)
    return response.content

# Gradio Interface
iface = gr.Interface(
    fn=study_buddy_ai,
    inputs=gr.Textbox(lines=2, placeholder="Ask your study question or request a study tip..."),
    outputs=gr.Markdown(),
    title="📚 Study Buddy AI",
    description="Ask academic questions or get study tips powered by Groq + Phi Agents!"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
