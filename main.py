import uvicorn
from server.api import app
from utils.config import ConfigManager
from core.brain import Brain
from core.speech import SpeechProcessor
import argparse

def run_api():
    """Run the FastAPI server"""
    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, reload=True)

def run_cli():
    """Run command-line interface"""
    config_manager = ConfigManager()
    brain = Brain(config_manager)
    speech = SpeechProcessor(config_manager)
    
    print(f"Starting {config_manager.config.name} Assistant...")
    print("Type 'exit' to quit, 'clear' to clear memory")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            brain.clear_memory()
            print("Memory cleared")
            continue
            
        response = brain.process_query(user_input)
        print(f"\nAssistant: {response}")
        
        # Speak the response
        speech.speak_text(response)

def run_voice():
    """Run voice-activated assistant"""
    config_manager = ConfigManager()
    brain = Brain(config_manager)
    speech = SpeechProcessor(config_manager)
    
    wake_word = config_manager.config.speech.wake_word
    print(f"Voice assistant started. Say '{wake_word}' to activate.")
    print("Press Ctrl+C to exit")
    
    import time
    while True:
        try:
            # Simple wake word detection (basic version)
            print("\nListening for wake word...")
            text = speech.listen_from_mic()
            
            if text and wake_word.lower() in text.lower():
                print(f"Wake word detected! Listening for command...")
                
                # Get the actual command (remove wake word)
                command = text.lower().replace(wake_word.lower(), "").strip()
                
                if command:
                    print(f"Command: {command}")
                    response = brain.process_query(command)
                    print(f"Response: {response}")
                    speech.speak_text(response)
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Assistant")
    parser.add_argument("--mode", choices=["api", "cli", "voice"], 
                       default="api", help="Run mode")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api()
    elif args.mode == "cli":
        run_cli()
    elif args.mode == "voice":
        run_voice()