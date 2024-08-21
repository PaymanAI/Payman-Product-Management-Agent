import os
import uuid
import requests
from typing import TypedDict, Literal, List, Optional, Dict, Any
from flask import Flask, request, jsonify
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dotenv import load_dotenv
load_dotenv()

# Environment variables and constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAYMAN_AGENT_ID = os.getenv("PAYMAN_AGENT_ID")
PAYMAN_API_SECRET = os.getenv("PAYMAN_API_SECRET")

HEADERS = {
    "x-payman-agent-id": PAYMAN_AGENT_ID,
    "x-payman-api-secret": PAYMAN_API_SECRET,
    "Content-Type": "application/json",
    "Accept": "application/vnd.payman.v1+json"
}

# Simple in-memory mapping for task_id to thread_id
task_thread_map = {}

@tool
def prioritize_product_roadmap(feedback: str) -> str:
    """
    Analyze the feedback and generate a list of product features that should be prioritized on the roadmap.
    The features will be classified as P0 (high priority), P1 (medium priority), and P2 (low priority).
    """
    print(f"Prioritizing roadmap with feedback: {feedback}")

    prompt = (
        f"Given the following feedback from users:\n\n"
        f"{feedback}\n\n"
        "Please analyze this feedback and suggest product features that should be added to the roadmap. "
        "Prioritize these features as P0 (high priority), P1 (medium priority), and P2 (low priority)."
    )
    
    # Use the model to generate the prioritized roadmap features
    response = model_without_tools.invoke(prompt)
    
    if isinstance(response, AIMessage):
        return response.content
    else:
        return "Error: Unable to generate prioritized product roadmap."


@tool
def generate_feedback_and_create_task(company_info: str, product_info: str, payout: int, assignTaskEmails: Optional[List[str]]) -> str:
    """Generate feedback questions and create a task with the generated questions."""
    
    # Step 1: Generate feedback questions
    print("Company Info:", company_info)
    print("Product Info:", product_info)
    
    prompt = (
        f"Based on the following company information:\n\n{company_info}\n\n"
        f"and the following product details:\n\n{product_info}\n\n"
        "Please generate a list of 5 to 7 structured feedback questions. Each question should be specific, focused on key aspects of the product, and formatted as follows:\n"
        "1. [Question focusing on the user's experience or understanding of the product]\n"
        "2. [Question asking for specific feedback on a feature or functionality]\n"
        "3. [Question asking about potential improvements or suggestions]\n"
        "4. [Question focusing on the impact or value of the product to the user]\n"
        "5. [Question assessing the ease of use or learning curve of the product]\n\n"
        "Ensure the questions are clear, concise, and relevant to the product’s goals."
    )
    
    # Use the model without tools to generate the questions
    response = model_without_tools.invoke([{"role": "user", "content": prompt}])
    
    print("Feedback Questions Response:", response)
    
    # Since `response` is a list of AIMessage objects, access the first message and get its content
    questions = response.content.strip().split('\n')
    task_description = "Please provide feedback on the following questions:\n" + "\n".join(questions)
    
    # Step 2: Create the task with the generated questions
    payload = {
        "title": "Provide Feedback on Task Completer Feature",
        "description": task_description,
        "payout": payout * 100,  # Convert payout to cents
        "currency": {
            "code": "USD"
        },
        "requiredSubmissions": 1,
        "submissionPolicy": "OPEN_SUBMISSIONS_ONE_PER_USER",
        "category": "MARKETING",
        "inviteEmails": assignTaskEmails
    }
    
    response = requests.post(
        "https://agent-sandbox.payman.ai/api/tasks", headers=HEADERS, json=payload
    )
    
    if response.status_code == 200:
        try:
            task_data = response.json()
            task_id = task_data['id']
            return f"Task created successfully. Task ID: {task_id}"
        except KeyError as e:
            return f"Error: {e}. The task could not be created because 'id' was not found in the response."
    else:
        return f"Failed to create task. Error: {response.text}"




# Initialize the OpenAI model
model = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY).bind_tools([generate_feedback_and_create_task, prioritize_product_roadmap])
model_without_tools = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Define the AgentState type
class AgentState(TypedDict):
    input_messages: List[Dict[str, str]]
    task_id: Optional[str]
    waiting_for_webhook: bool
    thread_id: str  # Include thread_id in the state

def initialize_state(input_messages: List[Dict[str, str]], thread_id: str) -> AgentState:
    return AgentState(
        input_messages=input_messages,
        task_id=None,
        waiting_for_webhook=False,
        thread_id=thread_id
    )


def call_llm(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    messages = state["input_messages"]
    response = model.invoke(messages)
    
    # Print the entire response for debugging
    print("LLM Response:", response)
    
    # Dynamically check if there are tool calls
    if 'tool_calls' in response.additional_kwargs and response.additional_kwargs['tool_calls']:
        # Dynamically create AIMessage with tool calls
        ai_message = AIMessage(
            content=response.content,  # Content can be empty if the focus is on tool calls
            additional_kwargs={'tool_calls': response.additional_kwargs['tool_calls']}
        )
        messages.append(ai_message)
    else:
        # Standard LLM response as AIMessage
        messages.append(AIMessage(
            content=response.content
        ))

    return {
        "input_messages": messages,  # Ensure input_messages is always updated
        **state  # Include all other state variables
    }


def run_tool(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    messages = state["input_messages"]
    last_message = messages[-1]

    print("last message", last_message)

    if not isinstance(last_message, AIMessage):
        print("Error: Last message is not an AIMessage.")
        return {"input_messages": messages}

    if not hasattr(last_message, 'additional_kwargs') or 'tool_calls' not in last_message.additional_kwargs:
        print("No tool calls found in the last message.")
        return {"input_messages": messages}

    print("Tool calls found. Running tool...")

    # Use ToolNode to execute the combined tool
    tool_node = ToolNode(tools=[generate_feedback_and_create_task])

    tool_input = {"messages": [last_message]}

    print("executed tool node")
    response = tool_node.invoke(tool_input)

    print("Tool Response:", response)

    # Extract the tool call ID and tool name
    tool_call_id = last_message.additional_kwargs['tool_calls'][0]['id']
    tool_name = last_message.additional_kwargs['tool_calls'][0]['function']['name']

    # Create a ToolMessage with the correct response
    tool_message = ToolMessage(
        content=response['messages'][0].content,
        name=tool_name,
        tool_call_id=tool_call_id
    )

    updates = {
        "input_messages": messages + [tool_message],
        "waiting_for_webhook": "Task created successfully" in response['messages'][0].content
    }

    if updates["waiting_for_webhook"]:
        task_id = tool_message.content.split("Task ID: ")[1].strip()
        task_thread_map[task_id] = state["thread_id"]
        updates["task_id"] = task_id

    return updates

def prioritize_roadmap(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Entering prioritize_roadmap function with state: {state}")
    
    descriptions = state.get("descriptions", [])
    feedback = "\n".join(descriptions)
    
    roadmap = prioritize_product_roadmap(feedback)
    
    roadmap_message = HumanMessage(content=f"Based on the feedback, here's the prioritized roadmap:\n\n{roadmap}")
    state["input_messages"].append(roadmap_message)
    
    state["perform_prioritization"] = False
    
    print(f"Exiting prioritize_roadmap function with updated state: {state}")
    return state

def route(state: AgentState) -> Literal["call_llm", "run_tool", "prioritize_roadmap", END]:
    print(f"Routing state: {state}")

    if state.get("perform_prioritization", False):
        print("Performing roadmap prioritization")
        return "prioritize_roadmap"
    
    if state.get("waiting_for_webhook", False):
        print("Waiting for webhook response, stopping graph.")
        return END
    
    # Check if a response has already been sent after webhook
    if state.get("response_sent_after_webhook", False):
        print("Response after webhook already sent, stopping graph.")
        return END
    
    last_message = state["input_messages"][-1]

    print("last message", last_message)
     # Debug print for state values
    print("Current 'descriptions':", state.get("descriptions", []))
    print("Current 'perform_prioritization':", state.get("perform_prioritization", False))
    print("Current 'waiting_for_webhook':", state.get("waiting_for_webhook", False))
    
    # Continue with normal routing logic
    if isinstance(last_message, HumanMessage) and last_message.content:
        print("true user")
        return "call_llm"
    elif isinstance(last_message, AIMessage) and hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls'):
        print("calling tool")
        return "run_tool"
    else:
        return "call_llm"

workflow = StateGraph(AgentState)

# Add the nodes to the workflow
workflow.add_node("call_llm", call_llm)
workflow.add_node("run_tool", run_tool)
workflow.add_node("prioritize_roadmap", prioritize_roadmap)  # New node for roadmap prioritization

# Set the entry point for the workflow
workflow.set_entry_point("call_llm")

# Define the conditional edges to transition between states
workflow.add_conditional_edges("call_llm", route)
workflow.add_conditional_edges("run_tool", route)
workflow.add_conditional_edges("prioritize_roadmap", route)  # Conditional edge for the new node

# Compile the graph
graph = workflow.compile()

memory = MemorySaver()

app = Flask(__name__)

def fetch_task_submissions(task_id: str) -> Optional[List[str]]:
    url = f"https://agent-sandbox.payman.ai/api/tasks/{task_id}/submissions"
    params = {"statuses": "APPROVED"}  
    response = requests.get(url, headers=HEADERS, params=params)
    

    if response.status_code == 200:
        submissions = response.json()
        
        # Adjust the logic here to correctly extract the descriptions
        if submissions and 'results' in submissions:
            descriptions = [submission['details'].get('description', 'No description provided') for submission in submissions['results']]
            return descriptions
    else:
        print(f"Failed to fetch submissions for task {task_id}. Error: {response.text}")
        return None



@app.route('/webhook/task_completed', methods=['POST'])
def task_completed_webhook():
    data = request.json
    print(f"Webhook Data Received: {data}")

    event_type = data.get('eventType')
    if event_type != 'task.completed':
        return jsonify({"message": "Event ignored"}), 200

    details = data.get('details')
    if not details:
        return jsonify({"error": "Missing details"}), 400

    task_id = details.get('task_id')
    if not task_id:
        return jsonify({"error": "Missing task_id"}), 400

    print(f"Task ID from webhook: {task_id}")
    print(f"Current task_thread_map: {task_thread_map}")

    thread_id = task_thread_map.get(task_id)
    if not thread_id:
        return jsonify({"error": "No matching task found"}), 404

    state = memory.storage.get(thread_id)
    if state is None:
        return jsonify({"error": "No state found for this thread_id"}), 404

    print(f"Waiting for a webhook: {state.get('waiting_for_webhook', True)}")

    if not state.get("waiting_for_webhook", False):
        return jsonify({"error": "Webhook received but not expected"}), 400

    print("Resuming conversation...")

    submissions = fetch_task_submissions(task_id)
    labeled_submissions = [f"Submission {i+1}: {s}" for i, s in enumerate(submissions, 1)]

    state["descriptions"] = labeled_submissions
    state["perform_prioritization"] = True
    state["waiting_for_webhook"] = False

    print(f"Updated state before prioritization: {state}")

    # Directly call prioritize_roadmap
    updated_state = prioritize_roadmap(state, {})

    # Then call LLM to generate a response based on the prioritized roadmap
    final_state = call_llm(updated_state, {})

    final_state["response_sent_after_webhook"] = True

    memory.storage[thread_id] = final_state

    print("Final Messages:")
    for msg in final_state["input_messages"]:
        if isinstance(msg, dict):
            print(msg.get("content", "No content"))
        elif hasattr(msg, "content"):
            print(msg.content)
        else:
            print(f"Unexpected message type: {type(msg)}")

    # Extract the prioritized roadmap
    prioritized_roadmap = next((msg.content for msg in reversed(final_state["input_messages"]) 
                                if isinstance(msg, HumanMessage) and "prioritized roadmap" in msg.content), 
                                "Prioritized roadmap not found")

    return jsonify({
        "message": "Task completed, roadmap prioritized, and conversation resumed",
        "thread_id": thread_id,
        "prioritized_roadmap": prioritized_roadmap,
        "final_messages": [msg.content if hasattr(msg, "content") else str(msg) for msg in final_state["input_messages"]]
    }), 200



@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    print("start_conversation called")
    data = request.json
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({"error": "Missing user_input"}), 400

    # Generate a unique thread_id for this conversation
    thread_id = str(uuid.uuid4())

    initial_state = initialize_state([{"role": "user", "content": user_input}], thread_id)
    run_id = str(uuid.uuid4())
    final_state = graph.invoke(initial_state)

    # Log the state being saved
    print(f"Saving state with thread_id: {thread_id} and run_id: {run_id}")
    memory.storage[thread_id] = final_state  # Explicitly saving state with thread_id as the key

    print(f"Conversation started with thread_id: {thread_id} and run_id: {run_id}")
    
    if final_state.get('waiting_for_webhook', False):
        return jsonify({
            "message": "Task created. Waiting for completion.",
            "run_id": run_id, 
            "task_id": final_state.get('task_id'),
            "thread_id": thread_id  # Return the thread_id
        }), 202
    else:
        return jsonify({
            "message": "Conversation completed.",
            "final_messages": [msg["content"] for msg in final_state["input_messages"]]
        }), 200


if __name__ == '__main__':
    app.run(debug=True)
