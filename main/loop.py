import os
import json
import requests
import time
from tqdm import tqdm
from collections import Counter, deque
from datetime import datetime
from rag_utils import RAGSystem

# groq api setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MEMORY_FILE = "memory.txt"


rag_system = RAGSystem()
    

# all the apush periods
AP_PERIODS = [
    "Period 1 (1491-1607): Native American Societies and European Exploration",
    "Period 2 (1607-1754): Colonial America",
    "Period 3 (1754-1800): The American Revolution",
    "Period 4 (1800-1848): Early Republic and Expansion",
    "Period 5 (1844-1877): Civil War and Reconstruction",
    "Period 6 (1865-1898): Industrialization and Gilded Age",
    "Period 7 (1890-1945): Progressive Era and World Wars",
    "Period 8 (1945-1980): Cold War and Civil Rights",
    "Period 9 (1980-Present): Modern America"
]

def log_llm_call(purpose: str, model: str, messages: list):
    """log info about llm api calls"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[LLM Call at {timestamp}]")
    print(f"Purpose: {purpose}")
    print(f"Model: {model}")
    print("Context:")
    for msg in messages:
        role = msg["role"]
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"- {role}: {content}")
    print("-" * 80)

def load_memory(file_path=MEMORY_FILE):
    """load existing memory from file"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return ""

def save_memory(memory_content, file_path=MEMORY_FILE):
    """save memory to file"""
    with open(file_path, 'w') as f:
        f.write(memory_content)

def query_groq(messages, model="llama3-8b-8192", max_completion_tokens=1000, purpose="Unknown"):
    """helper function to call groq api"""
    # log the llm call

    log_llm_call(purpose, model, messages)
   
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        response_json = response.json()
        
        if "error" in response_json:
            print(f"API Error: {response_json['error']}")
            return None
        if "choices" not in response_json or not response_json["choices"]:
            print(f"Unexpected API response: {response_json}")
            return None
            
        return response_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error processing response: {e}")
        return None

def generate_ap_question(period, topic=None, question_type="multiple_choice"):
    """generate an apush practice question"""
    topic_context = f" focusing on {topic}" if topic else ""
    prompt = f"""
    Generate an AP US History practice question for {period}{topic_context}
    
    Question type: {question_type}
    
    For multiple choice questions, include:
    1. The question
    2. Four possible answer options (A, B, C, D)
    3. The correct answer
    4. Brief explanation (2-3 sentences)
    5. Key historical context (1-2 sentences)
    6. AP exam relevance (1 sentence)
    
    Format the response as:
    QUESTION:
    [question text]
    
    OPTIONS:
    A) [first option]
    B) [second option]
    C) [third option]
    D) [fourth option]
    
    ANSWER:
    [correct answer letter]
    
    EXPLANATION:
    [brief explanation]
    
    HISTORICAL CONTEXT:
    [key context]
    
    AP RELEVANCE:
    [exam relevance]
    """
    
    messages = [
        {"role": "system", "content": "You are an AP US History expert creating exam-style questions"},
        {"role": "user", "content": prompt}
    ]
    
    return query_groq(messages, model="llama3-8b-8192", max_completion_tokens=500, purpose="Generate AP question")

def save_practice_problem(problem: str, period: str):
    """save practice problem to memory"""
    current_memory = load_memory()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # get key info from problem
    parts = problem.split('\n\n')
    question_text = ""
    correct_answer = ""
    
    for part in parts:
        if part.startswith("QUESTION:"):
            question_text = part.replace("QUESTION:", "").strip()
        elif part.startswith("ANSWER:"):
            correct_answer = part.replace("ANSWER:", "").strip()
    
    # check for learning patterns
    pattern_prompt = f"""
    Question: {question_text}
Period: {period}
    Correct Answer: {correct_answer}
    
    Identify if this practice question reveals any learning patterns or difficulties
    If yes, format as: "User has shown difficulty with: [specific concept/pattern]"
    If no clear pattern, return empty string
    """
    
    messages = [
        {"role": "system", "content": "You are an AP US History expert identifying learning patterns"},
        {"role": "user", "content": pattern_prompt}
    ]
    
    pattern = query_groq(messages, model="llama3-8b-8192", max_completion_tokens=150, purpose="Analyze practice problem pattern")
    if pattern:
        new_entry = f"\n[{timestamp}] {pattern}"
        updated_memory = current_memory + new_entry
        save_memory(updated_memory)

def get_relevant_practice_problems(period: str) -> str:
    """get relevant practice problems for current period"""
    memory = load_memory()
    if not memory:
        return ""
    
    relevance_prompt = f"""
    Current period: {period}
    
    Past practice problems and interactions:
    {memory}
    
    Based on the current period, which previous practice problems and learning patterns are most relevant?
    Provide a concise summary of relevant problems and learning patterns
    """
    
    messages = [
        {"role": "system", "content": "You are an AP US History expert identifying relevant practice problems and learning patterns"},
        {"role": "user", "content": relevance_prompt}
    ]
    
    return query_groq(messages, model="llama3-8b-8192", max_completion_tokens=150, purpose="Get relevant practice problems")

def show_periods():
    """show all apush periods"""
    print("\nAP US History Periods:")
    for i, period in enumerate(AP_PERIODS, 1):
        print(f"{i}. {period}")

def practice_mode():
    """interactive apush practice mode"""
    print("\n=== AP US History Practice Mode ===")
    print("Type 'exit' to return to main chat")
    print("Type 'periods' to see all AP periods")
    print("Type 'problems' to see all practice problems")
    
    while True:
        show_periods()
        period_choice = input("\nEnter the number of the period you want to practice (or 'exit'/'problems'): ").strip()
        
        if period_choice.lower() == 'exit':
            break
        elif period_choice.lower() == 'problems':
            print("\nAll Practice Problems:")
            print(load_memory())
            continue
        
        try:
            period_index = int(period_choice) - 1
            if 0 <= period_index < len(AP_PERIODS):
                period = AP_PERIODS[period_index]
            else:
                print("Invalid period number. Please try again.")
                continue
        except ValueError:
            print("Please enter a valid number.")
            continue
        
        # get topic from student
        print(f"\nYou've selected {period}")
        print("What specific topic would you like to practice?")
        print("Examples:")
        print("- Political developments")
        print("- Social movements")
        print("- Economic changes")
        print("- Key events")
        print("- Important figures")
        print("- Cultural changes")
        print("Or any other topic you're interested in!")
        
        selected_topic = input("\nEnter your topic (or press Enter for a general question): ").strip()
        
        # generate new question
        full_question = generate_ap_question(period, selected_topic)
        if not full_question:
            print("Sorry, I couldn't generate a practice question. Please try again.")
            continue
        
        # split question into parts
        parts = full_question.split('\n\n')
        question_text = ""
        solution = ""
        options = []
        correct_answer = ""
        
        for part in parts:
            if part.startswith("QUESTION:"):
                question_text = part.replace("QUESTION:", "").strip()
            elif part.startswith("OPTIONS:"):
                options_text = part.replace("OPTIONS:", "").strip()
                options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
            elif part.startswith("ANSWER:"):
                correct_answer = part.replace("ANSWER:", "").strip()
                solution += part + "\n\n"
            elif part.startswith("EXPLANATION:") or part.startswith("HISTORICAL CONTEXT:") or part.startswith("AP RELEVANCE:"):
                solution += part + "\n\n"
        
        # show question and options
        print("\nHere's your AP US History practice question:")
        print(question_text)
        if options:
            print("\nOptions:")
            for opt in options:
                print(opt)
        
        # get user's answer
        while True:
            attempt = input("\nEnter the letter of your answer (A, B, C, or D) (or 'skip' to see solution, 'hint' for a hint): ").strip().upper()
            
            if attempt.lower() == 'skip':
                print("\nCorrect Answer:", correct_answer)
                # update memory for skipped question
                update_memory(
                    question_text,
                    "Question skipped",
                    f"Student skipped question about {selected_topic if selected_topic else 'general topic'} in {period}"
                )
                break
            elif attempt.lower() == 'hint':
                # give a hint without giving away the answer
                hint_prompt = f"""
                AP US History Question: {question_text}
                Period: {period}
                Topic: {selected_topic if selected_topic else 'General'}
                
                Provide a brief, focused hint (1-2 sentences) that guides the student without giving away the solution
                """
                
                messages = [
                    {"role": "system", "content": "You are an AP US History teacher providing hints"},
                    {"role": "user", "content": hint_prompt}
                ]
                
                hint = query_groq(messages, model="llama3-8b-8192", max_completion_tokens=100, purpose="Generate hint")
                if hint:
                    print("\nHint:", hint)
                continue
            
            if attempt in ['A', 'B', 'C', 'D']:
                selected_option = next((opt for opt in options if opt.startswith(attempt + ")")), None)
                if selected_option:
                    break
                else:
                    print("Invalid option. Please enter A, B, C, or D")
            else:
                print("Please enter a valid letter (A, B, C, or D)")
        
        # give feedback
        feedback_prompt = f"""
        AP US History Question: {question_text}
        Period: {period}
        Topic: {selected_topic if selected_topic else 'General'}
        Student's selected option: {selected_option if 'selected_option' in locals() else 'skipped'}
        Correct answer: {correct_answer}
        
        Provide brief, focused feedback (2-3 sentences) that:
        1. Acknowledges what was correct (if anything)
        2. Points out one key area for improvement
        3. Includes one specific tip for AP exam success
        """
        
        messages = [
            {"role": "system", "content": "You are an AP US History teacher providing concise feedback"},
            {"role": "user", "content": feedback_prompt}
        ]
        
        feedback = query_groq(messages, model="llama3-8b-8192", max_completion_tokens=150, purpose="Generate feedback")
        if feedback:
            print("\nFeedback:", feedback)
            
            # update memory with the interaction
            is_correct = selected_option and selected_option.startswith(correct_answer)
            memory_context = f"Topic: {selected_topic if selected_topic else 'General'}, Period: {period}"
            update_memory(
                question_text,
                f"Selected: {selected_option}, Correct: {correct_answer}",
                f"{'Correct answer' if is_correct else 'Incorrect answer'} on {memory_context}. {feedback}"
            )
        
        # save the question
        save_practice_problem(full_question, period)

def get_relevant_memory(query: str) -> str:
    """get relevant past learnings from memory"""
    memory = load_memory()
    if not memory:
        return ""
    
    relevance_prompt = f"""
    User query: {query}
    Past interactions and practice: {memory}
    
    Identify any patterns of difficulty or learning gaps from past interactions that are relevant to this query
    Only include insights if they show a clear pattern of misunderstanding or difficulty
    Format as: "User has shown difficulty with: [specific concept/pattern]"
    If no relevant patterns found, return empty string
    """
    
    messages = [
        {"role": "system", "content": "You are an AP US History expert identifying learning patterns"},
        {"role": "user", "content": relevance_prompt}
    ]
    
    return query_groq(messages, model="llama3-8b-8192", max_completion_tokens=150, purpose="Get relevant memory")

def update_memory(question: str, response: str, feedback: str):
    """update memory with new interaction"""
    current_memory = load_memory()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # check for learning patterns
    pattern_prompt = f"""
    Question: {question}
    Response: {response}
    Feedback: {feedback}
    
    Identify if this interaction shows a clear pattern of difficulty or misunderstanding
    If yes, format as: "User has shown difficulty with: [specific concept/pattern]"
    If no clear pattern, return empty string
    """
    
    messages = [
        {"role": "system", "content": "You are an AP US History expert identifying learning patterns"},
        {"role": "user", "content": pattern_prompt}
    ]
    
    pattern = query_groq(messages, model="llama3-8b-8192", max_completion_tokens=150, purpose="Update memory with pattern")
    if pattern:
        new_entry = f"\n[{timestamp}] {pattern}"
        updated_memory = current_memory + new_entry
        save_memory(updated_memory)

def chat_with_memory():
    """main chat function with memory"""
    print("Welcome to AP US History Study Buddy!")
    print("\nCommands:")
    print("- Type 'exit' to quit")
    print("- Type 'memory' to see full memory")
    print("- Type 'practice' to enter practice mode")
    print("- Type 'periods' to see all AP periods")
    
    while True:
        try:
            # start conversation context
            conversation_context = []
            
            # get initial question
            question = input("\nYour question: ").strip()
            
            if question.lower() == 'exit':
                break
            elif question.lower() == 'memory':
                print("\nCurrent Memory:")
                print(load_memory())
                continue
            elif question.lower() == 'practice':
                practice_mode()
                continue
            elif question.lower() == 'periods':
                show_periods()
                continue
            
            # start conversation loop
            while True:
                # get relevant past learnings
                relevant_memory = get_relevant_memory(question)
                
                # prepare conversation
                messages = [
                    {"role": "system", "content": "You are an AP US History expert tutor helping students prepare for the AP exam"}
                ]
                
                if relevant_memory:
                    messages.append({
                        "role": "system",
                        "content": f"Relevant past learnings to consider:\n{relevant_memory}"
                    })
                
                messages.extend(conversation_context)
                messages.append({"role": "user", "content": question})
                
                # get response
                response = query_groq(messages, model="llama3-70b-8192", max_completion_tokens=1000, purpose="Main chat response")
                if response:
                    print("\nAI:", response)
                    
                    # update conversation context
                    conversation_context.append({"role": "user", "content": question})
                    conversation_context.append({"role": "assistant", "content": response})
                    
                    # check for follow-up
                    follow_up = input("\nDo you have any follow-up questions? (yes/no): ").strip().lower()
                    if follow_up != 'yes':
                        # end conversation
                        update_memory(question, response, "")
                        break
                    
                    # get follow-up question
                    question = input("\nYour follow-up question: ").strip()
                else:
                    print("Sorry, I encountered an error. Please try again.")
                    break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print("Sorry, an error occurred. Please try again.")

chat_with_memory()