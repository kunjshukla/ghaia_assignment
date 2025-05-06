from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import autogen
from typing import Dict, List, Optional, Tuple
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not GEMINI_API_KEY or not OPENROUTER_API_KEY:
    raise ValueError("Missing required environment variables: GEMINI_API_KEY and/or OPENROUTER_API_KEY")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configuration for the agents
config_list = [
    {
        "model": "gemini-2.0-flash",
        "api_key": GEMINI_API_KEY
    }
]

# System Memory Class
class SystemMemory:
    def __init__(self):
        self.requests = {}
        self.request_counter = 0
    
    def add_request(self, user_id: str, request_text: str) -> int:
        """Add a new request to memory and return request ID"""
        request_id = self.request_counter
        self.request_counter += 1
        
        self.requests[request_id] = {
            "user_id": user_id,
            "request_text": request_text,
            "timestamp": time.time(),
            "status": "new",
            "history": [{"agent": "system", "action": "request_received", "timestamp": time.time()}],
            "resolution": None
        }
        return request_id
    
    def update_request(self, request_id: int, agent: str, action: str, details: Optional[Dict] = None) -> None:
        """Update the request with new action information"""
        if request_id not in self.requests:
            raise ValueError(f"Request ID {request_id} not found in memory")
        
        self.requests[request_id]["history"].append({
            "agent": agent,
            "action": action,
            "details": details,
            "timestamp": time.time()
        })
    
    def set_status(self, request_id: int, status: str) -> None:
        """Update the status of a request"""
        if request_id not in self.requests:
            raise ValueError(f"Request ID {request_id} not found in memory")
        
        self.requests[request_id]["status"] = status
    
    def set_resolution(self, request_id: int, resolution: str) -> None:
        """Set the resolution for a request"""
        if request_id not in self.requests:
            raise ValueError(f"Request ID {request_id} not found in memory")
        
        self.requests[request_id]["resolution"] = resolution
        self.requests[request_id]["status"] = "resolved"
    
    def get_request(self, request_id: int) -> Dict:
        """Get request details"""
        if request_id not in self.requests:
            raise ValueError(f"Request ID {request_id} not found in memory")
        
        return self.requests[request_id]
    
    def get_similar_requests(self, keywords: List[str], limit: int = 3) -> List[Dict]:
        """Find similar past requests based on keywords"""
        similar_requests = []
        
        for req_id, request in self.requests.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in request["request_text"].lower():
                    score += 1
            
            if score > 0:
                similar_requests.append((req_id, request, score))
        
        similar_requests.sort(key=lambda x: x[2], reverse=True)
        return [{"id": req_id, "request": req["request_text"], "resolution": req["resolution"]} 
                for req_id, req, _ in similar_requests[:limit] if req["resolution"]]

# Initialize system memory
system_memory = SystemMemory()

# Helper Functions
def extract_keywords(text: str) -> List[str]:
    """Extract keywords from user request text"""
    common_stopwords = ["i", "my", "me", "our", "we", "is", "are", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with"]
    
    text = text.lower()
    for char in ",.?!;:()[]{}\"'":
        text = text.replace(char, " ")
    
    words = text.split()
    keywords = [word for word in words if word not in common_stopwords and len(word) > 2]
    
    categories = ["vpn", "email", "password", "printer", "software", "network", "computer", "laptop", "wifi", "internet"]
    for category in categories:
        if category in text.lower() and category not in keywords:
            keywords.append(category)
            
    return keywords

def generate_ai_solution(issue_description: str, keywords: List[str]) -> Tuple[bool, str]:
    """Generate a solution using the AI model"""
    prompt = f"""
    You are an IT support specialist. A user has reported the following issue:
    
    "{issue_description}"
    
    Keywords identified: {', '.join(keywords)}
    
    Please provide a step-by-step solution to this IT problem. Make your response concise yet thorough, 
    focusing on practical troubleshooting steps the user can take. If the issue likely requires 
    escalation to IT staff, mention that as well. Format your response as direct instructions to the user.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            return True, response.text.strip()
        return False, "I wasn't able to generate a specific solution for your issue. I recommend contacting IT support directly."
    
    except Exception as e:
        logger.error(f"Error generating AI solution: {e}")
        return False, f"Error generating solution: {str(e)}"

def reset_password(employee_id: str) -> str:
    """Simulate password reset function"""
    return f"Password reset link has been sent to the email associated with employee ID: {employee_id}"

# Custom Gemini LLM
class GeminiLLM:
    def __init__(self, model="gemini-2.0-flash", api_key=None):
        self.model = model
        if api_key:
            genai.configure(api_key=api_key)
        
    def create(self, messages, max_tokens=1024):
        """Convert messages to Gemini format and generate a response"""
        prompt = self._format_messages(messages)
        
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        
        model = genai.GenerativeModel(self.model)
        try:
            response = model.generate_content(prompt, generation_config=generation_config)
            return {"choices": [{"message": {"content": response.text}}]}
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return {"choices": [{"message": {"content": "I encountered an error processing your request."}}]}
    
    def _format_messages(self, messages):
        """Format AutoGen messages to a prompt Gemini can use"""
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
            else:
                formatted_prompt += f"{content}\n\n"
                
        return formatted_prompt.strip()

# Agent System Messages
user_intake_system_message = """
You are the User Intake Agent for an AI-powered IT help desk system. Your job is to:
1. Greet the user and collect their issue description
2. Extract key details about their problem
3. Identify relevant keywords that categorize their issue
4. Pass the processed information to the appropriate agent

Be professional, courteous, and thorough in gathering information.
"""

resolution_system_message = """
You are the Resolution Agent for an AI-powered IT help desk system. Your job is to:
1. Analyze user issues based on keywords and details
2. Generate tailored solutions using AI for the specific problem
3. Provide step-by-step resolution instructions
4. Execute functions like password resets when needed
5. If you cannot resolve the issue, escalate to the Escalation Agent

Be clear, precise, and helpful in your instructions. Draw from your knowledge of common IT issues.
"""

escalation_system_message = """
You are the Escalation Agent for an AI-powered IT help desk system. Your job is to:
1. Handle complex cases that couldn't be resolved automatically
2. Generate a structured summary of the issue for human IT support
3. Collect additional information that might help human agents resolve the issue
4. Provide the user with a ticket number and estimated response time

Be empathetic while maintaining professionalism.
"""

master_system_message = """
You are the Master Agent coordinating an AI-powered IT help desk system. Your job is to:
1. Receive user requests and direct them to the User Intake Agent
2. Manage communication between all agents
3. Track request status in the system memory
4. Ensure appropriate responses are delivered back to the user
5. Look for similar past resolved issues to speed up resolution

Maintain a professional, efficient communication style.
"""

llm_config = {
    "config_list": [
        {
            "model": "anthropic/claude-3-opus",
            "api_key": OPENROUTER_API_KEY,
            "base_url": "https://openrouter.ai/api/v1"
        }
    ]
}

# Initialize Agents
try:
    user_intake_agent = autogen.AssistantAgent(
        name="User_Intake_Agent",
        system_message=user_intake_system_message,
        llm_config=llm_config
    )

    resolution_agent = autogen.AssistantAgent(
        name="Resolution_Agent",
        system_message=resolution_system_message,
        llm_config=llm_config
    )

    escalation_agent = autogen.AssistantAgent(
        name="Escalation_Agent",
        system_message=escalation_system_message,
        llm_config=llm_config
    )

    master_agent = autogen.AssistantAgent(
        name="Master_Agent",
        system_message=master_system_message,
        llm_config=llm_config
    )

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )
except Exception as e:
    logger.error(f"Error initializing agents: {str(e)}")
    raise

# Group Chat Setup
groupchat = autogen.GroupChat(
    agents=[user_proxy, master_agent, user_intake_agent, resolution_agent, escalation_agent],
    messages=[],
    max_round=50
)

manager = autogen.GroupChatManager(groupchat=groupchat)

# Main Help Desk Function
def run_help_desk(user_id: str, user_request: str) -> str:
    """Run the help desk system for a user request"""
    try:
        if not user_id or user_id == "anonymous":
            return "Please provide a valid Employee ID to process your request."
        
        # Step 1: Master agent receives the request
        logger.info(f"Received IT help request from user {user_id}")
        
        # Step 2: User Intake Agent processes the request
        intake_result = process_intake(user_id, user_request)
        logger.info(f"Extracted keywords: {intake_result['keywords']}")
        
        # Step 3: Resolution Agent attempts to resolve
        resolution_result = attempt_resolution(
            request_id=intake_result['request_id'],
            keywords=intake_result['keywords'],
            issue_description=intake_result['issue_description']
        )
        
        # Step 4: Handle resolution or escalation
        if resolution_result['status'] == 'resolved':
            response = f"Here's a solution to your issue:\n\n{resolution_result['solution']}"
            
            if resolution_result['similar_issues']:
                response += "\n\nI noticed similar issues were reported before. Here's what worked previously:"
                for idx, issue in enumerate(resolution_result['similar_issues'], 1):
                    response += f"\n{idx}. Issue: {issue['request']}\n   Solution: {issue['resolution']}"
            
            return response
        else:
            # Step 5: Escalation Agent handles complex cases
            escalation_result = escalate_issue(
                request_id=intake_result['request_id'],
                issue_description=intake_result['issue_description'],
                keywords=intake_result['keywords']
            )
            
            # Generate general troubleshooting steps
            general_steps_prompt = f"""
            You are an IT support specialist. A user has the following issue:
            
            "{intake_result['issue_description']}"
            
            Keywords identified: {', '.join(intake_result['keywords'])}
            
            The issue is being escalated to IT staff, but in the meantime, provide 3-5 general troubleshooting 
            steps the user can try while waiting. Keep them simple, safe, and unlikely to make the situation worse.
            """
            
            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(general_steps_prompt)
                general_steps = response.text.strip() if response and hasattr(response, 'text') else "No general steps available."
            except Exception as e:
                logger.error(f"Error generating general steps: {e}")
                general_steps = "1. Restart your device\n2. Check your network connection\n3. Make sure all cables are properly connected"
            
            response = f"I wasn't able to automatically resolve your issue. I've escalated this to our IT support team.\n\n"
            response += f"Ticket number: {escalation_result['ticket_number']}\n"
            response += f"Estimated response time: {escalation_result['estimated_response_time']}\n\n"
            response += "In the meantime, here are some general troubleshooting steps you can try:\n"
            response += general_steps
            
            return response
    except Exception as e:
        logger.error(f"Error in run_help_desk: {str(e)}")
        return "An error occurred while processing your request. Please try again later."

# Agent Functions
def process_intake(user_id: str, issue_description: str) -> Dict:
    """Process the user intake and return structured information"""
    try:
        request_id = system_memory.add_request(user_id, issue_description)
        keywords = extract_keywords(issue_description)
        
        system_memory.update_request(
            request_id=request_id,
            agent="User_Intake_Agent",
            action="processed_request",
            details={"keywords": keywords}
        )
        
        return {
            "request_id": request_id,
            "user_id": user_id,
            "issue_description": issue_description,
            "keywords": keywords
        }
    except Exception as e:
        logger.error(f"Error in process_intake: {str(e)}")
        raise

def attempt_resolution(request_id: int, keywords: List[str], issue_description: str) -> Dict:
    """Attempt to resolve the issue using AI-generated solution"""
    try:
        similar_requests = system_memory.get_similar_requests(keywords)
        success, solution = generate_ai_solution(issue_description, keywords)
        
        if success:
            system_memory.update_request(
                request_id=request_id,
                agent="Resolution_Agent",
                action="generated_solution",
                details={"solution": solution}
            )
            
            system_memory.set_resolution(request_id, solution)
            
            return {
                "status": "resolved",
                "solution": solution,
                "similar_issues": similar_requests
            }
        else:
            system_memory.update_request(
                request_id=request_id,
                agent="Resolution_Agent",
                action="failed_to_generate_solution",
                details={"error": solution}
            )
            
            return {
                "status": "unresolved",
                "message": "Unable to generate solution",
                "similar_issues": similar_requests
            }
    except Exception as e:
        logger.error(f"Error in attempt_resolution: {str(e)}")
        raise

def escalate_issue(request_id: int, issue_description: str, keywords: List[str], attempted_solutions: List[str] = None) -> Dict:
    """Escalate the issue to human IT support"""
    try:
        ticket_number = f"IT-{request_id}-{int(time.time())}"
        
        issue_summary_prompt = f"""
        You are an IT support specialist. Summarize the following user issue for IT staff:
        
        "{issue_description}"
        
        Keywords identified: {', '.join(keywords)}
        
        Create a concise technical summary that would help IT staff understand the issue quickly.
        Include likely technical causes and any information that the staff should gather from the user.
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(issue_summary_prompt)
            technical_summary = response.text.strip() if response and hasattr(response, 'text') else "No technical summary available."
        except Exception as e:
            logger.error(f"Error generating technical summary: {e}")
            technical_summary = f"Error generating technical summary: {str(e)}"
        
        summary = {
            "ticket_number": ticket_number,
            "user_id": system_memory.get_request(request_id)["user_id"],
            "issue_description": issue_description,
            "technical_summary": technical_summary,
            "keywords": keywords,
            "attempted_solutions": attempted_solutions or [],
            "priority": "medium",
            "estimated_response_time": "4 hours"
        }
        
        system_memory.update_request(
            request_id=request_id,
            agent="Escalation_Agent",
            action="escalated",
            details=summary
        )
        
        system_memory.set_status(request_id, "escalated")
        
        return summary
    except Exception as e:
        logger.error(f"Error in escalate_issue: {str(e)}")
        raise

if __name__ == "__main__":
    user_id = "emp12345"
    user_request = "My VPN is not working. I tried to connect but it keeps failing with a timeout error."
    
    response = run_help_desk(user_id, user_request)
    print("\nFinal response to user:")
    print(response)