# Payman Product Management Agent

This Python application uses Flask to create a webhook endpoint for managing product feedback and prioritization tasks. The agent automates the process of gathering user feedback on product features and generating prioritized roadmaps based on that feedback.

## Key Features

- Automatically generates relevant feedback questions based on the company and feature description
- Creates tasks to collect user feedback with specified payouts
- Processes completed feedback to create prioritized product roadmaps
  
## Prerequisites

- Python 3.8+
- pip
- ngrok (for local development)

## Setup

**1. Clone the repository:**

```
git clone `https://github.com/yourusername/PAYMAN_PRODUCT_MANAGEMENT_AGENT.git`
```

```
cd PAYMAN_PRODUCT_MANAGEMENT_AGENT
```

**2. Create a virtual environment and activate it:**
```
python -m venv venv
```

*Mac use*

```
source venv/bin/activate
```

*Windows use*

```
venv\Scripts\activate
```

**3. Install the required packages:**
```
pip install -r requirements.txt
```

**4. Create a `.env` file in the root directory and add the following environment variables:**
```
OPENAI_API_KEY=your_openai_api_key
PAYMAN_AGENT_ID=your_payman_agent_id
PAYMAN_API_SECRET=your_payman_api_secret
```

## Running the Application

**1. Start the Flask server:**
```
python agent.py
```
   
**2. In a separate terminal, start ngrok to create a public URL for your local server:**
```
ngrok http 5000
```
   
**3. Copy the ngrok URL (it should look like `https://something.ngrok.io`).**

**4. In the Payman agent settings, set the webhook URL to:**
```
https://{your-ngrok-url}/webhook/task_completed
```

> [!NOTE]
> Make sure your AI Agent has funds in test mode to spend

## Usage

**1. To start a conversation and create a feedback task, use the following curl command (replace the user_input with your specific details):**

```
curl -X POST http://127.0.0.1:5000/start_conversation 
-H "Content-Type: application/json" 
-d '{
"user_input": "Company Payman enables AI agents to pay humans for completing tasks. Our latest feature, Task Completer, streamlines task completion and payment for users, ensuring efficient and accurate task handling. Please generate feedback questions to gather user insights on the Task Completer feature. Create tasks to collect feedback from users with a payout of $200 for each completed feedback task. No specific emails should be assigned to these tasks."
}'
```

## Note

Remember to keep your `.env` file secure and never commit it to version control. Add `.env` to your `.gitignore` file to prevent accidental commits.

