# Payman Product Management Agent

This Python application uses Flask to create a webhook endpoint for managing product feedback and prioritization tasks.

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

The application will now listen for webhook events from Payman. When a task is completed, it will process the feedback and generate a prioritized product roadmap.

## Note

Remember to keep your `.env` file secure and never commit it to version control. Add `.env` to your `.gitignore` file to prevent accidental commits.

