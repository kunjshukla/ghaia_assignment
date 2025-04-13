from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import autogen
from typing import Dict, List, Optional, Tuple
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os


load_dotenv()


# Configuration for Google Gemini
# You would need to set up your API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Replace with actual API key in production
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
2
# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configuration for the agents
config_list = [
    {
        "model": "gemini-pro",  # Using Gemini Pro model
        "api_key": GEMINI_API_KEY
    }
]

# Knowledge base for common IT issues
KNOWLEDGE_BASE = {
    "vpn": {
        "not connecting": "Try these steps: 1) Restart your VPN client 2) Check your internet connection 3) Ensure your credentials are correct",
        "slow": "VPN slowness can be caused by: network congestion, server distance, or bandwidth limitations. Try connecting to a different server."
    },
    "email": {
        "cannot access": "Try these steps: 1) Verify your internet connection 2) Clear browser cache 3) Reset your email password",
        "not receiving": "Check your spam folder, verify your account storage isn't full, ensure email forwarding isn't enabled incorrectly."
    },
    "password": {
        "reset": "I can help reset your password. Please verify your employee ID for security purposes."
    },
    "printer": {
        "not printing": "Check if: 1) Printer is powered on 2) Connected to network 3) Has paper and ink 4) Is set as default printer"
    },
    "software": {
        "installation": "For software installation requests, I'll need: 1) Software name 2) Version 3) Business justification"
    }
}

# Create a class to maintain system memory
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
        
        # Sort by score and return top results
        similar_requests.sort(key=lambda x: x[2], reverse=True)
        return [{"id": req_id, "request": req["request_text"], "resolution": req["resolution"]} 
                for req_id, req, _ in similar_requests[:limit] if req["resolution"]]

# Initialize system memory
system_memory = SystemMemory()

# Helper functions for the agents
def extract_keywords(text: str) -> List[str]:
    """Extract keywords from user request text"""
    # This is a simple implementation - in production, you'd use NLP techniques
    common_stopwords = ["i", "my", "me", "our", "we", "is", "are", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with"]
    
    # Clean and split text
    text = text.lower()
    for char in ",.?!;:()[]{}\"'":
        text = text.replace(char, " ")
    
    words = text.split()
    keywords = [word for word in words if word not in common_stopwords and len(word) > 2]
    
    # Add common IT categories if they appear in the text
    categories = ["vpn", "email", "password", "printer", "software", "network", "computer", "laptop", "wifi", "internet"]
    for category in categories:
        if category in text.lower() and category not in keywords:
            keywords.append(category)
            
    return keywords

def find_knowledge_base_solution(keywords: List[str]) -> Tuple[bool, str]:
    """Search the knowledge base for solutions based on keywords"""
    for keyword in keywords:
        if keyword in KNOWLEDGE_BASE:
            # Search for specific issue within the category
            for issue, solution in KNOWLEDGE_BASE[keyword].items():
                if any(k in issue for k in keywords):
                    return True, solution
            
            # If no specific issue matched but category exists, return the first solution
            first_issue = list(KNOWLEDGE_BASE[keyword].keys())[0]
            return True, KNOWLEDGE_BASE[keyword][first_issue]
    
    return False, "No solution found in knowledge base"

def reset_password(employee_id: str) -> str:
    """Simulate password reset function"""
    # In a real system, this would integrate with your ID management system
    return f"Password reset link has been sent to the email associated with employee ID: {employee_id}"

# Custom AutoGen-compatible LLM class for Gemini
class GeminiLLM:
    def __init__(self, model="gemini-pro", api_key=None):
        self.model = model
        # Initialize the model
        if api_key:
            genai.configure(api_key=api_key)
        
    def create(self, messages, max_tokens=1024):
        """Convert messages to Gemini format and generate a response"""
        # Process messages to format compatible with Gemini
        prompt = self._format_messages(messages)
        
        # Generate content
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
            print(f"Error generating content: {e}")
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

# Define the agents
user_intake_system_message = """
You are the User Intake Agent for an IT help desk system. Your job is to:
1. Greet the user and collect their issue description
2. Extract key details about their problem
3. Identify relevant keywords that categorize their issue
4. Pass the processed information to the appropriate agent

Be professional, courteous, and thorough in gathering information.
"""

resolution_system_message = """
You are the Resolution Agent for an IT help desk system. Your job is to:
1. Analyze user issues based on keywords and details
2. Search the knowledge base for solutions
3. Provide step-by-step resolution instructions when available
4. Execute functions like password resets when needed
5. If you cannot resolve the issue, escalate to the Escalation Agent

Be clear, precise, and helpful in your instructions.
"""

escalation_system_message = """
You are the Escalation Agent for an IT help desk system. Your job is to:
1. Handle complex cases that couldn't be resolved automatically
2. Generate a structured summary of the issue for human IT support
3. Collect additional information that might help human agents resolve the issue
4. Provide the user with a ticket number and estimated response time

Be empathetic while maintaining professionalism.
"""

master_system_message = """
You are the Master Agent coordinating an IT help desk system. Your job is to:
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
            "model": "anthropic/claude-3-opus",  # You can change this to gpt-4/gemini/etc.
            "api_key": OPENROUTER_API_KEY,
            "base_url": "https://openrouter.ai/api/v1"
        }
    ]
}

# Create the agents
user_intake_agent = autogen.AssistantAgent(
    name="User_Intake_Agent",
    system_message=user_intake_system_message,
    llm_config= llm_config
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

# Define functions for agents
def process_intake(user_id: str, issue_description: str) -> Dict:
    """Process the user intake and return structured information"""
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

def attempt_resolution(request_id: int, keywords: List[str], issue_description: str) -> Dict:
    """Attempt to resolve the issue using the knowledge base"""
    # Look for similar past resolved issues
    similar_requests = system_memory.get_similar_requests(keywords)
    
    # Check if solution exists in knowledge base
    found, solution = find_knowledge_base_solution(keywords)
    
    if found:
        system_memory.update_request(
            request_id=request_id,
            agent="Resolution_Agent",
            action="found_solution",
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
            action="no_solution_found",
            details={}
        )
        
        return {
            "status": "unresolved",
            "message": "No direct solution found in knowledge base",
            "similar_issues": similar_requests
        }

def escalate_issue(request_id: int, issue_description: str, keywords: List[str], attempted_solutions: List[str] = None) -> Dict:
    """Escalate the issue to human IT support"""
    # Generate a ticket number (in real system, would come from ticketing system)
    ticket_number = f"IT-{request_id}-{int(time.time())}"
    
    summary = {
        "ticket_number": ticket_number,
        "user_id": system_memory.get_request(request_id)["user_id"],
        "issue_description": issue_description,
        "keywords": keywords,
        "attempted_solutions": attempted_solutions or [],
        "priority": "medium",  # Default priority, could be determined by keywords
        "estimated_response_time": "4 hours"  # Default SLA
    }
    
    system_memory.update_request(
        request_id=request_id,
        agent="Escalation_Agent",
        action="escalated",
        details=summary
    )
    
    system_memory.set_status(request_id, "escalated")
    
    return summary

# Set up the groupchat for agents to collaborate
groupchat = autogen.GroupChat(
    agents=[user_proxy, master_agent, user_intake_agent, resolution_agent, escalation_agent],
    messages=[],
    max_round=50
)

manager = autogen.GroupChatManager(groupchat=groupchat)

# Main help desk function
def run_help_desk(user_id: str, user_request: str):
    """Run the help desk system for a user request"""
    # Step 1: Master agent receives the request
    master_response = f"Received IT help request from user {user_id}. Processing..."
    print(f"Master Agent: {master_response}")
    
    # Step 2: User Intake Agent processes the request
    print(f"User Intake Agent: Analyzing request...")
    intake_result = process_intake(user_id, user_request)
    print(f"User Intake Agent: Extracted keywords: {intake_result['keywords']}")
    
    # Step 3: Resolution Agent attempts to resolve
    print(f"Resolution Agent: Searching for solutions...")
    resolution_result = attempt_resolution(
        request_id=intake_result['request_id'],
        keywords=intake_result['keywords'],
        issue_description=intake_result['issue_description']
    )
    
    # Step 4: Handle resolution or escalation
    if resolution_result['status'] == 'resolved':
        print(f"Resolution Agent: Issue resolved!")
        response = f"I found a solution to your issue:\n\n{resolution_result['solution']}"
        
        # Mention similar issues if available
        if resolution_result['similar_issues']:
            response += "\n\nI noticed similar issues were reported before. Here's what worked previously:"
            for idx, issue in enumerate(resolution_result['similar_issues'], 1):
                response += f"\n{idx}. Issue: {issue['request']}\n   Solution: {issue['resolution']}"
                
        print(f"Master Agent: {response}")
        return response
    else:
        print(f"Resolution Agent: Could not resolve issue. Escalating...")
        
        # Step 5: Escalation Agent handles complex cases
        escalation_result = escalate_issue(
            request_id=intake_result['request_id'],
            issue_description=intake_result['issue_description'],
            keywords=intake_result['keywords']
        )
        
        response = f"I wasn't able to automatically resolve your VPN issue. I've escalated this to our IT support team.\n\n"
        response += f"Ticket number: {escalation_result['ticket_number']}\n"
        response += f"Estimated response time: {escalation_result['estimated_response_time']}\n\n"
        response += "In the meantime, here are some general troubleshooting steps for VPN issues:\n"
        response += "1. Restart your computer and try connecting again\n"
        response += "2. Check if you can access other websites to confirm your internet connection is working\n"
        response += "3. Make sure you're using the latest version of the VPN client\n"
        
        print(f"Escalation Agent: {response}")
        return response

# Example usage
if __name__ == "__main__":
    user_id = "emp12345"
    user_request = "My VPN is not working. I tried to connect but it keeps failing with a timeout error."
    
    response = run_help_desk(user_id, user_request)
    print("\nFinal response to user:")
    print(response)