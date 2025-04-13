import streamlit as st
import time
from help_desk import run_help_desk, system_memory


user_id = "anonymous"


def main():
    # Set up page config
    st.set_page_config(
        page_title="AI Help Desk",
        page_icon="🤖",
        layout="wide",
    )
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # if "user_id" not in st.session_state:
    #     st.session_state.user_id = ""
    
    if "current_request_id" not in st.session_state:
        st.session_state.current_request_id = None
    
    # App header
    st.title("🤖 AI IT Help Desk")
    st.markdown("### Your AI-powered IT support assistant")
    
    # User ID input
    with st.sidebar:
        # st.header("User Information")
        # user_id = st.text_input("Employee ID", value=st.session_state.user_id)
        
        # if user_id and user_id != st.session_state.user_id:
        #     st.session_state.user_id = user_id
        #     st.success(f"Welcome, Employee #{user_id}!")
        
        st.markdown("---")
        st.header("Common Issues")
        
        # Add quick links for common issues
        if st.button("VPN Connection Problems"):
            handle_quick_issue("My VPN is not connecting properly.")
        
        if st.button("Email Access Issues"):
            handle_quick_issue("I cannot access my email account.")
        
        if st.button("Password Reset"):
            handle_quick_issue("I need to reset my password.")
        
        if st.button("Printer Not Working"):
            handle_quick_issue("My printer is not printing documents.")
        
        st.markdown("---")
        
        # Add ticket tracking section
        st.header("Ticket Tracking")
        
        if len(system_memory.requests) > 0:
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
    
    # Display ticket details if selected
    if st.session_state.current_request_id is not None:
        display_ticket_details(st.session_state.current_request_id)
    
    # Chat interface
    st.markdown("---")
    st.header("Submit Your IT Request")
    
    # Display chat history
    chat_container = st.container(height=400) # Added height
    with chat_container:
            for message in st.session_state.chat_history:
            # Use message['role'] directly for st.chat_message
                avatar = "🤖" if message.get("role") == "assistant" else None # Use .get
                role = message.get("role", "unknown") # Use .get
                with st.chat_message(role, avatar=avatar):
                    st.markdown(message.get("content", "*No content*")) # Use .get
    
    # User input
    user_input = st.chat_input("Describe your IT issue here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Display user message immediately (optional but good UX)
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Show thinking indicator
        with st.spinner("Processing your request..."):
            # Process the request
            response = run_help_desk(user_id, user_input)
            
            # Add system response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Set current request ID to latest
            if system_memory.requests:
                st.session_state.current_request_id = max(system_memory.requests.keys())
        
        # Force refresh to show new messages
        st.rerun()

def handle_quick_issue(issue_text):
    """Handle clicking a quick issue button"""
    if st.session_state.user_id:
        # Add the issue to chat history
        st.session_state.chat_history.append({"role": "user", "content": issue_text})
        
        # Process the request
        with st.spinner("Processing your request..."):
            response = run_help_desk(st.session_state.user_id, issue_text)
            
            # Add system response to chat history
            st.session_state.chat_history.append({"role": "system", "content": response})
            
            # Set current request ID to latest
            if system_memory.requests:
                st.session_state.current_request_id = max(system_memory.requests.keys())
        
        # Force refresh
        st.rerun()
    else:
        st.sidebar.error("Please enter your Employee ID first.")

def display_ticket_details(request_id):
    """Display detailed information for a specific ticket"""
    # Get request details
    request_data = system_memory.get_request(request_id)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Ticket #{request_id} Details")
    
    # Create status indicator
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
    
    # Display history in an expander
    with st.sidebar.expander("Event History"):
        for idx, event in enumerate(request_data["history"], 1):
            event_time = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
            st.markdown(f"**{idx}.** [{event_time}] {event['agent']}: {event['action']}")
            if event.get('details'):
                details = event['details']
                for key, value in details.items():
                    if isinstance(value, list) and key == "keywords":
                        st.markdown(f"- **{key}**: {', '.join(value)}")
                    elif not isinstance(value, dict) and not isinstance(value, list):
                        st.markdown(f"- **{key}**: {value}")
                    else:
                        st.markdown(f"- **{key}**: [Complex Data]")
                        

if __name__ == "__main__":
    main()