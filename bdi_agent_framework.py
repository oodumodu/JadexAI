import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from openai import OpenAI  # or your preferred LLM client

# Constants
DEFAULT_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONFIDENCE = 1.0
DEFAULT_PRIORITY = 1


@dataclass
class Belief:
    """Represents an agent's belief with confidence level."""
    key: str
    value: Any
    confidence: float = DEFAULT_CONFIDENCE


@dataclass
class Desire:
    """Represents an agent's goal with priority and context."""
    goal: str
    priority: int = DEFAULT_PRIORITY
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Intention:
    """Represents an agent's intention to perform an action."""
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None


class BDIAgent:
    """
    A BDI (Belief-Desire-Intention) agent that uses an LLM for reasoning.
    
    The agent maintains beliefs about the world, desires (goals), and intentions
    (planned actions) to achieve those goals.
    """
    
    def __init__(self, name: str, llm_client: OpenAI, system_prompt: str = ""):
        """
        Initialize a BDI agent.
        
        Args:
            name: The agent's name
            llm_client: OpenAI client instance
            system_prompt: Custom system prompt for the agent
        """
        self.name = name
        self.llm = llm_client
        self.beliefs: Dict[str, Belief] = {}
        self.desires: List[Desire] = []
        self.intentions: List[Intention] = []
        self.system_prompt = (
            system_prompt or 
            f"You are {name}, a BDI agent. Reason about beliefs, desires, and intentions."
        )
        
    def add_belief(self, key: str, value: Any, confidence: float = DEFAULT_CONFIDENCE) -> None:
        """
        Add or update a belief.
        
        Args:
            key: Belief identifier
            value: Belief content
            confidence: Confidence level (0.0-1.0)
        """
        self.beliefs[key] = Belief(key, value, confidence)
    
    def add_desire(self, goal: str, priority: int = DEFAULT_PRIORITY, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new desire/goal.
        
        Args:
            goal: Description of the desired goal
            priority: Priority level (higher = more important)
            context: Additional context for the goal
        """
        self.desires.append(Desire(goal, priority, context or {}))
    
    def get_context(self) -> str:
        """
        Generate context string for LLM containing current agent state.
        
        Returns:
            Formatted string with beliefs, desires, and intentions
        """
        beliefs_str = json.dumps(
            {k: asdict(v) for k, v in self.beliefs.items()}, 
            indent=2
        )
        desires_str = json.dumps(
            [asdict(d) for d in self.desires], 
            indent=2
        )
        intentions_str = json.dumps(
            [asdict(i) for i in self.intentions], 
            indent=2
        )
        
        return f"""
Current State:
BELIEFS: {beliefs_str}
DESIRES: {desires_str}
CURRENT INTENTIONS: {intentions_str}
        """
    
    def reason(self, perception: str = "") -> List[Intention]:
        """
        Main reasoning cycle using LLM to determine intentions.
        
        Args:
            perception: New information or observations
            
        Returns:
            List of new intentions to execute
        """
        prompt = self._build_reasoning_prompt(perception)
        
        try:
            response = self.llm.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=DEFAULT_TEMPERATURE
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Update beliefs based on reasoning
            self._update_beliefs(result.get("belief_updates", []))
            
            # Update intentions
            new_intentions = self._create_intentions(result.get("new_intentions", []))
            self.intentions = new_intentions
            
            self._log_reasoning(result.get("reasoning", "No explanation provided"))
            
            return self.intentions
            
        except json.JSONDecodeError as e:
            print(f"[{self.name}] Error parsing LLM response: {e}")
            return []
        except Exception as e:
            print(f"[{self.name}] Error in reasoning: {e}")
            return []
    
    def _build_reasoning_prompt(self, perception: str) -> str:
        """Build the reasoning prompt for the LLM."""
        return f"""{self.system_prompt}

{self.get_context()}

NEW PERCEPTION: {perception}

Based on your current beliefs, desires, and any new perception, decide what intentions/actions to take.
Consider:
1. Which desires are most important given current beliefs?
2. What actions would help achieve these desires?
3. Should any current intentions be modified or cancelled?

Respond with a JSON object containing:
- "belief_updates": [{{"key": "string", "value": "any", "confidence": 0.0-1.0}}]
- "new_intentions": [{{"action": "string", "parameters": {{}}, "deadline": timestamp_or_null}}]
- "reasoning": "explanation of your decision process"

Only include belief updates if perceptions change your understanding.
"""
    
    def _update_beliefs(self, belief_updates: List[Dict[str, Any]]) -> None:
        """Update agent beliefs based on reasoning results."""
        for belief_update in belief_updates:
            self.add_belief(
                belief_update["key"], 
                belief_update["value"], 
                belief_update.get("confidence", DEFAULT_CONFIDENCE)
            )
    
    def _create_intentions(self, intention_data_list: List[Dict[str, Any]]) -> List[Intention]:
        """Create intention objects from reasoning results."""
        intentions = []
        for intention_data in intention_data_list:
            intention = Intention(
                action=intention_data["action"],
                parameters=intention_data.get("parameters", {}),
                deadline=intention_data.get("deadline")
            )
            intentions.append(intention)
        return intentions
    
    def _log_reasoning(self, reasoning: str) -> None:
        """Log the agent's reasoning process."""
        print(f"[{self.name}] Reasoning: {reasoning}")
    
    def execute_intentions(self) -> List[str]:
        """
        Execute current intentions.
        
        Returns:
            List of execution results
        """
        results = []
        current_time = time.time()
        
        # Use list copy to avoid modification during iteration
        for intention in self.intentions[:]:
            # Check if intention has expired
            if intention.deadline:
                try:
                    deadline = float(intention.deadline) if isinstance(intention.deadline, str) else intention.deadline
                    if current_time > deadline:
                        self.intentions.remove(intention)
                        results.append(f"EXPIRED: {intention.action}")
                        continue
                except (ValueError, TypeError):
                    # If deadline can't be converted to float, treat as not expired
                    pass
            
            # Execute action
            result = self.execute_action(intention.action, intention.parameters)
            results.append(f"EXECUTED: {intention.action} -> {result}")
            
            # Remove completed intention
            self.intentions.remove(intention)
        
        return results
    
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> str:
        """
        Execute a specific action. Override this for domain-specific actions.
        
        Args:
            action: Action to execute
            parameters: Action parameters
            
        Returns:
            Result of action execution
        """
        return f"Action '{action}' with params {parameters} completed"
    
    def cycle(self, perception: str = "") -> Dict[str, Any]:
        """
        Complete BDI cycle: perceive -> reason -> act.
        
        Args:
            perception: New information or observations
            
        Returns:
            Dictionary containing cycle results and agent state
        """
        intentions = self.reason(perception)
        execution_results = self.execute_intentions()
        
        return {
            "intentions_formed": len(intentions),
            "actions_executed": execution_results,
            "current_beliefs": list(self.beliefs.keys()),
            "active_desires": len(self.desires)
        }


def display_agent_state(agent: BDIAgent) -> None:
    """Display the current state of the agent."""
    print("\n" + "="*50)
    print("CURRENT AGENT STATE")
    print("="*50)
    print(f"Agent: {agent.name}")
    print(f"Beliefs: {len(agent.beliefs)}")
    for key, belief in agent.beliefs.items():
        print(f"  - {key}: {belief.value} (confidence: {belief.confidence})")
    print(f"Desires: {len(agent.desires)}")
    for i, desire in enumerate(agent.desires):
        print(f"  - {desire.goal} (priority: {desire.priority})")
    print(f"Active Intentions: {len(agent.intentions)}")
    for i, intention in enumerate(agent.intentions):
        print(f"  - {intention.action} (params: {intention.parameters})")
    print("="*50)


def interactive_session() -> None:
    """Run an interactive session with the BDI agent."""
    # Initialize LLM client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Create agent
    agent = BDIAgent(
        "TaskBot", 
        client, 
        "You are a helpful task management agent. You want to be efficient and helpful."
    )
    
    # Add initial beliefs and desires
    agent.add_belief("current_time", "09:00", 0.9)
    agent.add_belief("energy_level", "high", 0.8)
    agent.add_desire("complete_project", priority=1)
    agent.add_desire("take_break", priority=2)
    
    print("🤖 BDI Agent Interactive Session")
    print("Welcome! I'm your BDI agent. I can help you manage tasks and make decisions.")
    print("\nCommands:")
    print("  - Just type a message to interact with me")
    print("  - 'add belief <key> <value> [confidence]' - Add a new belief")
    print("  - 'add desire <goal> [priority]' - Add a new desire")
    print("  - 'show state' - Display current agent state")
    print("  - 'quit' or 'exit' - End the session")
    print("-" * 50)
    
    display_agent_state(agent)
    
    while True:
        try:
            user_input = input("\n🗣️  You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye! Session ended.")
                break
            
            elif user_input.lower() == 'show state':
                display_agent_state(agent)
                continue
            
            elif user_input.lower().startswith('add belief'):
                parts = user_input.split(maxsplit=3)
                if len(parts) >= 4:
                    key = parts[2]
                    value = parts[3]
                    confidence = 1.0
                    if len(parts) > 4:
                        try:
                            confidence = float(parts[4])
                        except ValueError:
                            confidence = 1.0
                    agent.add_belief(key, value, confidence)
                    print(f"✅ Added belief: {key} = {value} (confidence: {confidence})")
                else:
                    print("❌ Format: add belief <key> <value> [confidence]")
                continue
            
            elif user_input.lower().startswith('add desire'):
                parts = user_input.split(maxsplit=3)
                if len(parts) >= 3:
                    goal = parts[2]
                    priority = 1
                    if len(parts) > 3:
                        try:
                            priority = int(parts[3])
                        except ValueError:
                            priority = 1
                    agent.add_desire(goal, priority)
                    print(f"✅ Added desire: {goal} (priority: {priority})")
                else:
                    print("❌ Format: add desire <goal> [priority]")
                continue
            
            elif user_input == '':
                continue
            
            # Process user input through BDI cycle
            print("\n🧠 Processing...")
            result = agent.cycle(user_input)
            
            print(f"\n🤖 Agent Response:")
            print(f"   Intentions formed: {result['intentions_formed']}")
            print(f"   Actions executed: {len(result['actions_executed'])}")
            if result['actions_executed']:
                for action in result['actions_executed']:
                    print(f"     - {action}")
            print(f"   Current beliefs: {len(result['current_beliefs'])}")
            print(f"   Active desires: {result['active_desires']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")


def main() -> None:
    """Main entry point - choose between interactive session or example run."""
    print("Choose mode:")
    print("1. Interactive session (recommended)")
    print("2. Run example scenarios")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_session()
        elif choice == "2":
            run_examples()
        else:
            print("Invalid choice. Starting interactive session...")
            interactive_session()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


def run_examples() -> None:
    """Run example scenarios (original functionality)."""
    # Initialize LLM client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Create agent
    agent = BDIAgent(
        "TaskBot", 
        client, 
        "You are a helpful task management agent. You want to be efficient and helpful."
    )
    
    # Add initial beliefs and desires
    agent.add_belief("current_time", "09:00", 0.9)
    agent.add_belief("energy_level", "high", 0.8)
    agent.add_desire("complete_project", priority=1)
    agent.add_desire("take_break", priority=2)
    
    scenarios = [
        "User says: I need to finish my presentation by 2 PM",
        "User says: I'm feeling tired and need a break",
        "User says: I have a meeting in 30 minutes",
        "User says: The project deadline changed to tomorrow"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*20} SCENARIO {i} {'='*20}")
        print(f"Input: {scenario}")
        result = agent.cycle(scenario)
        print("Result:")
        print(json.dumps(result, indent=2))
        print("-" * 50)


if __name__ == "__main__":
    main()