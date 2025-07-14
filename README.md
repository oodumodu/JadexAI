# JadexAI - BDI Agent Framework

A Python implementation of a **Belief-Desire-Intention (BDI)** agent framework that leverages Large Language Models (LLMs) for intelligent reasoning and decision-making.

## 🚀 Features

- **BDI Architecture**: Complete implementation of the Belief-Desire-Intention paradigm
- **LLM Integration**: Uses OpenAI's GPT models for intelligent reasoning
- **Interactive Mode**: Real-time conversation with your BDI agent
- **Flexible Design**: Easy to extend with custom actions and behaviors
- **State Management**: Persistent tracking of beliefs, desires, and intentions
- **Confidence Levels**: Beliefs with associated confidence scores
- **Priority System**: Prioritized goal management
- **Deadline Support**: Time-aware intention execution

## 🏗️ Architecture

### Core Components

- **Beliefs**: Agent's knowledge about the world with confidence levels
- **Desires**: Goals the agent wants to achieve with priority levels
- **Intentions**: Planned actions to achieve desires with optional deadlines

### BDI Cycle

1. **Perceive**: Process new information from the environment
2. **Reason**: Use LLM to determine appropriate intentions based on beliefs and desires
3. **Act**: Execute planned intentions and update agent state

## 📦 Installation

### Prerequisites

- Python 3.7+
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/oodumodu/JadexAI.git
   cd JadexAI
   ```

2. **Install dependencies**
   ```bash
   pip install openai
   ```

3. **Set up your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🎯 Quick Start

### Interactive Mode (Recommended)

```bash
python bdi_agent_framework.py
```

Choose option **1** for interactive mode and start conversing with your BDI agent!

### Example Interaction

```
🗣️  You: I need to finish my presentation by 2 PM
🧠 Processing...
🤖 Agent Response:
   Intentions formed: 2
   Actions executed: 2
     - EXECUTED: prioritize_presentation_task -> Action completed
     - EXECUTED: set_deadline_reminder -> Action completed
```

## 📚 Usage Examples

### Basic Agent Creation

```python
from openai import OpenAI
from bdi_agent_framework import BDIAgent

# Initialize LLM client
client = OpenAI(api_key="your-api-key")

# Create agent
agent = BDIAgent(
    name="TaskBot",
    llm_client=client,
    system_prompt="You are a helpful task management agent."
)

# Add initial beliefs and desires
agent.add_belief("current_time", "09:00", confidence=0.9)
agent.add_belief("energy_level", "high", confidence=0.8)
agent.add_desire("complete_project", priority=1)
agent.add_desire("take_break", priority=2)

# Run a BDI cycle
result = agent.cycle("I have a meeting in 30 minutes")
print(result)
```

### Custom Actions

```python
class CustomAgent(BDIAgent):
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> str:
        if action == "send_email":
            recipient = parameters.get("recipient")
            subject = parameters.get("subject")
            # Your email sending logic here
            return f"Email sent to {recipient} with subject: {subject}"
        elif action == "schedule_meeting":
            time = parameters.get("time")
            # Your meeting scheduling logic here
            return f"Meeting scheduled for {time}"
        else:
            return super().execute_action(action, parameters)
```

## 🛠️ API Reference

### BDIAgent Class

#### Constructor
```python
BDIAgent(name: str, llm_client: OpenAI, system_prompt: str = "")
```

#### Key Methods

- `add_belief(key: str, value: Any, confidence: float = 1.0)` - Add or update a belief
- `add_desire(goal: str, priority: int = 1, context: Dict = None)` - Add a new goal
- `reason(perception: str = "")` - Main reasoning cycle using LLM
- `execute_intentions()` - Execute all current intentions
- `cycle(perception: str = "")` - Complete BDI cycle: perceive → reason → act

### Data Classes

#### Belief
```python
@dataclass
class Belief:
    key: str
    value: Any
    confidence: float = 1.0
```

#### Desire
```python
@dataclass
class Desire:
    goal: str
    priority: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
```

#### Intention
```python
@dataclass
class Intention:
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
```

## 🎮 Interactive Commands

When running in interactive mode, you can use these commands:

- **Regular conversation**: Just type your message
- `add belief <key> <value> [confidence]` - Add a new belief
- `add desire <goal> [priority]` - Add a new desire
- `show state` - Display current agent state
- `quit` or `exit` - End the session

## ⚙️ Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)

### Constants

You can modify these constants in the code:

```python
DEFAULT_MODEL = "gpt-4"           # OpenAI model to use
DEFAULT_TEMPERATURE = 0.7        # LLM temperature
DEFAULT_CONFIDENCE = 1.0         # Default belief confidence
DEFAULT_PRIORITY = 1             # Default desire priority
```

## 🔧 Customization

### Extending the Framework

1. **Custom Actions**: Override `execute_action()` method
2. **Custom Reasoning**: Modify `_build_reasoning_prompt()` method  
3. **Custom Beliefs**: Add domain-specific belief types
4. **Custom Desires**: Implement specialized goal types

### Example: Calendar Agent

```python
class CalendarAgent(BDIAgent):
    def __init__(self, name: str, llm_client: OpenAI):
        super().__init__(name, llm_client, 
                        "You are a calendar management agent. Help users schedule and manage appointments.")
    
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> str:
        if action == "schedule_appointment":
            return self._schedule_appointment(parameters)
        elif action == "cancel_appointment":
            return self._cancel_appointment(parameters)
        # Add more calendar-specific actions
        return super().execute_action(action, parameters)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the BDI architecture from multi-agent systems research
- Built with OpenAI's GPT models for intelligent reasoning
- Thanks to the Python community for excellent libraries and tools

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/oodumodu/JadexAI/issues) page
2. Create a new issue with detailed description
3. Join our discussions in the [Discussions](https://github.com/oodumodu/JadexAI/discussions) tab

---

**Obi Odumodu** 