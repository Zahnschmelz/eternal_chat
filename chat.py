import json
import os
import subprocess
from datetime import datetime
from typing import List, Dict, Any

import openai
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

# --- Konfiguration & Konstanten ---
CONFIG_FILE = "config.json"
HISTORY_FILE = "chat_history.json"
MEMORY_FILE = "longterm_memory.json"

console = Console()

BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BROW = "\033[38;5;94m"
PURP = "\033[38;5;135m"
ORANGE = "\033[38;5;208m"
LIGHT_GREY = "\033[37m"
DARK_GRAY = "\033[90m"
BOLD_RED = "\033[38;5;88m"

class ChatAssistant:
    def __init__(self):
        self.config = self.load_config()
        self.history = self.load_history()
        self.memory = self.load_memory()
        self.client = openai.OpenAI(base_url=self.config["url"], api_key="lm-studio")
        self.token_threshold = self.config["token_threshold"]
        self.summary = ""

    def load_config(self):
        default_config = {"url": "http://localhost:1234/v1", "token_threshold": 5000, "model": "local-model"}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return {**default_config, **json.load(f)}
        return default_config

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=4)

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        return []

    def save_memory(self, fact: str):
        self.memory.append(fact)
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=4)

    # --- Tools ---
    def tool_read_file(self, path: str):
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def tool_write_file(self, path: str, content: str):
        try:
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def tool_bash_command(self, command: str):
        try:
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
            return result
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output}"

    def tool_save_memory(self, fact: str):
        self.save_memory(fact)
        return "Fact saved successfully."

    def tool_get_memory(self):
        return f"Current memories: {', '.join(self.memory)}" if self.memory else "Memory is empty."

    def tool_load_all_memory(self):
        return f"All memories: {json.dumps(self.memory)}"

    def ask_user_for_tool(self, func_name: str, args: dict) -> str:
        arg_str = ", ".join([f"{k}={v}" for k, v in args.items()])
        console.print(f"\n[bold cyan]Tool Call:[/bold cyan] `{func_name}({arg_str})`")
        choice = Prompt.ask("Execute this tool?", choices=["y", "n", "m"], default="y")

        if choice == "y":
            if func_name == "read_file": return self.tool_read_file(args["path"])
            if func_name == "write_file": return self.tool_write_file(args["path"], args["content"])
            if func_name == "bash_command": return self.tool_bash_command(args["command"])
            if func_name == "save_memory": return self.tool_save_memory(args["fact"])
            if func_name == "get_memory": return self.tool_get_memory()
            if func_name == "load_all_memory": return self.tool_load_all_memory()
            return ""
        elif choice == "m":
            return Prompt.ask(f"Enter manual return value for `{func_name}`")
        else:
            return f"[Skipped {func_name}]"

    # --- Logik ---
    def get_token_count(self):
        return sum(len(m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])) for m in self.history if m['role'] != 'system') // 4

    def shrink_history(self):
        if len(self.history) <= 2: return
        dialogue = [m for m in self.history if m['role'] in ['user', 'assistant']]
        if len(dialogue) <= 2: return
        to_summarize = dialogue[:-2]
        remaining = dialogue[-2:]
        prompt = f"Summarize the following conversation history into a single concise paragraph: {json.dumps(to_summarize, default=str)}"
        response = self.client.chat.completions.create(model=self.config["model"], messages=[{"role": "user", "content": prompt}])
        self.summary = response.choices[0].message.content
        self.history = [{"role": "system", "content": f"Summary of previous chat: {self.summary}"}] + remaining

    def run_chat(self):
        session = PromptSession()
        custom_style = Style.from_dict({'prompt_color': 'cyan'})
        commands = ["/exit", "/clear", "/history", "/memory", "/shrink", "/threshold", "/tools", "/tokens", "/config", "/url", "/help"]
        completer = WordCompleter(commands)
        console.print(Panel("[bold green]Eternal AI-Chat[/bold green]\nType /help for commands."))


        while True:
            try:
                #user_input = session.prompt("\033[36m> \033[0m", completer=completer).strip()
                print()
                user_input = session.prompt(
                    HTML('<prompt_color>> </prompt_color>'),
                    completer=completer,
                    style=custom_style
                ).strip()
                if not user_input: continue
                if user_input == "/exit":
                    self.save_history()
                    break
                elif user_input == "/clear":
                    self.history = []
                    self.summary = ""
                    console.print("[yellow]History cleared.[/yellow]")
                elif user_input == "/history":
                    for m in self.history:
                        role_color = "bold cyan" if m['role'] == 'user' else "bold magenta"
                        content = m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])
                        console.print(f"[{role_color}]{m['role'].upper()}[/ {role_color}]: {content}")
                elif user_input == "/memory":
                    console.print(f"[blue]Memories:[/blue]\n{self.memory}")
                elif user_input == "/shrink":
                    self.shrink_history()
                    console.print("[green]History compressed.[/green]")
                elif user_input.startswith("/threshold"):
                    self.token_threshold = int(user_input.split()[1])
                    self.config["token_threshold"] = self.token_threshold
                    self.save_config()
                elif user_input == "/tokens":
                    console.print(f"Current estimated tokens: {self.get_token_count()}")
                elif user_input == "/config":
                    console.print(self.config)
                elif user_input.startswith("/url"):
                    self.config["url"] = user_input.split()[1]
                    self.client = openai.OpenAI(base_url=self.config["url"], api_key="lm-studio")
                    self.save_config()
                elif user_input == "/tools":
                    console.print("Available tools: `read_file`, `write_file`, `bash_command`, `save_memory`, `get_memory`, `load_all_memory`")
                elif user_input == "/help":
                    console.print("Commands: /exit, /clear, /history, /memory, /shrink, /threshold <n>, /tools, /tokens, /config, /url <url>, /help")
                else:
                    print()
                    self.history.append({"role": "user", "content": user_input})
                    if self.get_token_count() > self.token_threshold:
                        self.shrink_history()
                    self.process_message()
                    self.save_history()

            except KeyboardInterrupt: break
            except Exception as e: console.print(f"[red]Error: {e}[/red]")

    def process_message(self):
        now = datetime.now().strftime("%H:%M:%S")
        sys_msg = (
            "You are a professional system assistant. "
            "You can read/write files, run bash commands, and manage a long-term memory. "
            "Automatically store relevant facts in your long-term memory. "
            f"Current time: {now}. "
            "Use tools if needed. Work efficiently."
        )

        tools = [
            {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "bash_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "save_memory", "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}}},
            {"type": "function", "function": {"name": "get_memory", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "load_all_memory", "parameters": {"type": "object", "properties": {}}}}
        ]

        current_messages = [{"role": "system", "content": sys_msg}] + self.history

        with console.status("[bold blue]Thinking...", spinner="dots") as status:
            while True:
                response = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=current_messages,
                    tools=tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message

                if not msg.tool_calls:
                    if msg.content:
                        self.history.append({"role": "assistant", "content": msg.content})
                        console.print(Markdown(msg.content))
                    break

                assistant_msg_dict = {
                    "role": "assistant",
                    "content": msg.content if msg.content else "",
                    "tool_calls": [
                        {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                }

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    result = self.ask_user_for_tool(func_name, args)

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }

                    current_messages.append(tool_msg)

if __name__ == "__main__":
    assistant = ChatAssistant()
    assistant.run_chat()
