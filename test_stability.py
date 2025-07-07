#!/usr/bin/env python3
"""
Stability test for ShopGuard backend server
Tests memory management, error handling, and graceful degradation
"""

import time
import json
import threading
from unittest.mock import patch, MagicMock
import os

# Set test environment variables to avoid crashes
os.environ.update({
    "VIVO_APP_ID": "test_app_id",
    "VIVO_APP_KEY": "test_app_key", 
    "RAG_API_DOMAIN": "test.domain.com",
    "RAG_API_URI": "/test/uri",
    "CONVERSATION_TTL_SECONDS": "10",  # Short TTL for testing
    "MAX_CONVERSATIONS": "5",  # Low limit for testing
    "ENVIRONMENT": "test"
})

# Import after setting environment variables
import newserver

class TestStability:
    
    def test_conversation_history_cleanup(self):
        """Test that conversation history is properly cleaned up"""
        # Clear existing conversations
        newserver.conversation_history.clear()
        newserver.conversation_timestamps.clear()
        
        # Add some test conversations with old timestamps
        old_time = time.time() - 20  # 20 seconds ago (past TTL)
        current_time = time.time()
        
        newserver.conversation_history["old_user"] = [{"role": "user", "content": "old message"}]
        newserver.conversation_timestamps["old_user"] = old_time
        
        newserver.conversation_history["new_user"] = [{"role": "user", "content": "new message"}]
        newserver.conversation_timestamps["new_user"] = current_time
        
        # Force cleanup
        newserver.last_cleanup_time = 0  # Force cleanup to run
        newserver.cleanup_conversation_history()
        
        # Old user should be removed, new user should remain
        assert "old_user" not in newserver.conversation_history
        assert "new_user" in newserver.conversation_history
        
    def test_max_conversations_limit(self):
        """Test that conversation count is limited"""
        # Clear and add conversations beyond limit
        newserver.conversation_history.clear()
        newserver.conversation_timestamps.clear()
        
        current_time = time.time()
        
        # Add more conversations than the limit
        for i in range(10):  # More than MAX_CONVERSATIONS (5)
            user_id = f"user_{i}"
            newserver.conversation_history[user_id] = [{"role": "user", "content": f"message {i}"}]
            newserver.conversation_timestamps[user_id] = current_time - i  # Different timestamps
        
        # Force cleanup
        newserver.last_cleanup_time = 0
        newserver.cleanup_conversation_history()
        
        # Should be limited to MAX_CONVERSATIONS
        assert len(newserver.conversation_history) <= newserver.MAX_CONVERSATIONS
        
    def test_graceful_degradation_missing_rag(self):
        """Test that server handles missing RAG configuration gracefully"""
        # The server should have started without crashing even with missing configs
        assert newserver.rag_system_instance is None
        
        # Health check should still work
        assert hasattr(newserver, 'app')
        
    def test_memory_not_growing_indefinitely(self):
        """Test that memory usage doesn't grow indefinitely"""
        initial_count = len(newserver.conversation_history)
        
        # Simulate many users over time
        base_time = time.time()
        for i in range(20):
            user_id = f"test_user_{i}"
            # Simulate old conversations
            newserver.conversation_history[user_id] = [{"role": "user", "content": f"test {i}"}]
            newserver.conversation_timestamps[user_id] = base_time - (i * 2)  # Space them out in time
        
        # Force cleanup
        newserver.last_cleanup_time = 0
        newserver.cleanup_conversation_history()
        
        # Should not exceed limits
        assert len(newserver.conversation_history) <= newserver.MAX_CONVERSATIONS
        
    def test_health_check_provides_monitoring_data(self):
        """Test that health check provides useful monitoring information"""
        # Import FastAPI test client
        from fastapi.testclient import TestClient
        
        client = TestClient(newserver.app)
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for essential monitoring fields
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "services" in data
        assert "session_management" in data
        assert "configuration" in data
        
        # Check session management info
        session_info = data["session_management"]
        assert "active_sessions" in session_info
        assert "max_sessions_limit" in session_info
        assert "session_ttl_seconds" in session_info
        
    def test_timeout_configurations_exist(self):
        """Test that timeout configurations are properly set"""
        assert hasattr(newserver, 'DEFAULT_REQUEST_TIMEOUT')
        assert hasattr(newserver, 'LLM_REQUEST_TIMEOUT')
        assert hasattr(newserver, 'RAG_REQUEST_TIMEOUT')
        assert hasattr(newserver, 'SEARCH_REQUEST_TIMEOUT')
        
        # Should be reasonable values
        assert newserver.DEFAULT_REQUEST_TIMEOUT > 0
        assert newserver.LLM_REQUEST_TIMEOUT > 0
        assert newserver.RAG_REQUEST_TIMEOUT > 0
        assert newserver.SEARCH_REQUEST_TIMEOUT > 0

if __name__ == "__main__":
    # Run the tests
    test = TestStability()
    
    print("Testing conversation history cleanup...")
    test.test_conversation_history_cleanup()
    print("✓ Conversation history cleanup works")
    
    print("Testing max conversations limit...")
    test.test_max_conversations_limit()
    print("✓ Max conversations limit works")
    
    print("Testing graceful degradation...")
    test.test_graceful_degradation_missing_rag()
    print("✓ Graceful degradation works")
    
    print("Testing memory management...")
    test.test_memory_not_growing_indefinitely()
    print("✓ Memory management works")
    
    print("Testing health check...")
    test.test_health_check_provides_monitoring_data()
    print("✓ Health check provides monitoring data")
    
    print("Testing timeout configurations...")
    test.test_timeout_configurations_exist()
    print("✓ Timeout configurations exist")
    
    print("\nAll stability tests passed! ✓")