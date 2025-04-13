# Test script for AI Help Desk with Gemini
from help_desk import run_help_desk, system_memory

def test_vpn_issue():
    print("=" * 50)
    print("TEST CASE: VPN Connection Issue")
    print("=" * 50)
    
    user_id = "emp12345"
    user_request = "My VPN is not working. I tried to connect but it keeps failing with a timeout error."
    
    print(f"User ({user_id}): {user_request}")
    print("-" * 50)
    
    response = run_help_desk(user_id, user_request)
    
    print("\nFinal Response to User:")
    print(response)
    
    # Retrieve the request from memory to show how it's tracked
    request_id = 0  # First request will have ID 0
    request_data = system_memory.get_request(request_id)
    
    print("\nRequest tracking in system memory:")
    print(f"Request ID: {request_id}")
    print(f"Status: {request_data['status']}")
    print(f"History trail:")
    for idx, event in enumerate(request_data['history'], 1):
        print(f"  {idx}. Agent: {event['agent']}, Action: {event['action']}, Time: {time.ctime(event['timestamp'])}")

def test_password_reset():
    print("\n" + "=" * 50)
    print("TEST CASE: Password Reset Request")
    print("=" * 50)
    
    user_id = "emp54321"
    user_request = "I need to reset my email password. I've been locked out after too many attempts."
    
    print(f"User ({user_id}): {user_request}")
    print("-" * 50)
    
    response = run_help_desk(user_id, user_request)
    
    print("\nFinal Response to User:")
    print(response)
    
    # Retrieve the request from memory
    request_id = 1  # Second request will have ID 1
    request_data = system_memory.get_request(request_id)
    
    print("\nRequest tracking in system memory:")
    print(f"Request ID: {request_id}")
    print(f"Status: {request_data['status']}")
    print(f"History trail:")
    for idx, event in enumerate(request_data['history'], 1):
        print(f"  {idx}. Agent: {event['agent']}, Action: {event['action']}, Time: {time.ctime(event['timestamp'])}")

if __name__ == "__main__":
    import time
    test_vpn_issue()
    test_password_reset()