import streamlit as st
import time 
from typing import Optional
import logging
from help_desk import system_memory, run_help_desk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set up page config
    st.set_page_config(
        page_title="AI Help Desk",
        page_icon="🤖",
        layout="wide",
    )
    
    # Initialize session state with defaults
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "anonymous"
    if "current_request_id" not in st.session_state:
        st.session_state.current_request_id = None
    
    # App header
    st.title("🤖 AI IT Help Desk")
    st.markdown("### Your AI-powered IT support assistant")
    
    # Sidebar layout
    with st.sidebar:
        st.header("User Information")
        user_id = st.text_input("Employee ID", value=st.session_state.user_id)
        
        if user_id and user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
            st.success(f"Welcome, Employee #{user_id}!")
        
        st.markdown("---")
        st.header("Common Issues")
        
        # Quick issue buttons
        quick_issues = {
            "VPN Connection Problems": "My VPN is not connecting properly.",
            "Email Access Issues": "I cannot access my email account.",
            "Password Reset": "I need to reset my password.",
            "Printer Not Working": "My printer is not printing documents."
        }
        
        for button_label, issue_text in quick_issues.items():
            if st.button(button_label):
                handle_quick_issue(issue_text, user_id)
        
        st.markdown("---")
        
        # Ticket tracking
        st.header("Ticket Tracking")
        if system_memory.requests:
            st.subheader("Recent Requests")
            for req_id, req_data in system_memory.requests.items():
                status_color = {
                    "new": "🔵",
                    "in progress": "🟠",
                    "resolved": "🟢",
                    "escalated": "🟣"
                }.get(req_data["status"], "⚪")
                
                st.markdown(f"{status_color} **Ticket #{req_id}**: {req_data['status'].title()}")
                
                if st.button(f"View Details #{req_id}", key=f"details_{req_id}"):
                    st.session_state.current_request_id = req_id
        else:
            st.info("No tickets yet. Submit your first IT request!")
    
    # Display ticket details
    if st.session_state.current_request_id is not None:
        display_ticket_details(st.session_state.current_request_id)
    
    # Chat interface
    st.markdown("---")
    st.header("Submit Your IT Request")
    
    # Chat history container
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            role = message.get("role", "user")
            avatar = "🤖" if role == "assistant" else None
            with st.chat_message(role, avatar=avatar):
                st.markdown(message.get("content", "*No content*"))
    
    # User input
    user_input = st.chat_input("Describe your IT issue here...")
    
    if user_input:
        try:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
            
            # Process the request
            with st.spinner("Processing your request..."):
                response = run_help_desk(st.session_state.user_id, user_input)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Update current request ID
                if system_memory.requests:
                    st.session_state.current_request_id = max(system_memory.requests.keys())
            
            st.rerun()
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            st.error("An error occurred while processing your request. Please try again.")

def handle_quick_issue(issue_text: str, user_id: str) -> None:
    """Handle clicking a quick issue button"""
    try:
        if not user_id or user_id == "anonymous":
            st.sidebar.error("Please enter a valid Employee ID first.")
            return
        
        # Add the issue to chat history
        st.session_state.chat_history.append({"role": "user", "content": issue_text})
        
        # Process the request
        with st.spinner("Processing your request..."):
            response = run_help_desk(user_id, issue_text)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Update current request ID
            if system_memory.requests:
                st.session_state.current_request_id = max(system_memory.requests.keys())
        
        st.rerun()
    except Exception as e:
        logger.error(f"Error handling quick issue: {str(e)}")
        st.sidebar.error("An error occurred while processing your request.")

def display_ticket_details(request_id: int) -> None:
    """Display detailed information for a specific ticket"""
    try:
        request_data = system_memory.get_request(request_id)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"Ticket #{request_id} Details")
        
        status_color = {
            "new": "🔵",
            "in progress": "🟠",
            "resolved": "🟢",
            "escalated": "🟣"
        }.get(request_data["status"], "⚪")
        
        st.sidebar.markdown(f"**Status**: {status_color} {request_data['status'].title()}")
        st.sidebar.markdown(f"**Opened**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_data['timestamp']))}")
        st.sidebar.markdown(f"**Request**: {request_data['request_text']}")
        
        if request_data["resolution"]:
            st.sidebar.markdown(f"**Resolution**: {request_data['resolution']}")
        
        with st.sidebar.expander("Event History"):
            for idx, event in enumerate(request_data["history"], 1):
                event_time = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
                st.markdown(f"**{idx}.** [{event_time}] {event['agent']}: {event['action']}")
                if event.get('details'):
                    for key, value in event['details'].items():
                        if isinstance(value, list) and key == "keywords":
                            st.markdown(f"- **{key}**: {', '.join(value)}")
                        elif not isinstance(value, dict) and not isinstance(value, list):
                            st.markdown(f"- **{key}**: {value}")
                        else:
                            st.markdown(f"- **{key}**: [Complex Data]")
    except Exception as e:
        logger.error(f"Error displaying ticket details: {str(e)}")
        st.sidebar.error("Error retrieving ticket details.")

if __name__ == "__main__":
    main()