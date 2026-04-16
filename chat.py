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
import os
import re
import subprocess
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import datetime

# --- KONFIGURATION ---
HISTORY_FILE = "chat_history.json"
MEMORY_FILE = "memory.json"
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "token_threshold": 5000,
    "base_url": "http://127.0.0.1:1234/v1",
    "api_key": "lm-studio"
}
BASE_SYSTEM_PROMPT = (
    "You are a professional system assistant. "
    "You can read/write files, run bash commands, and manage a long-term memory. "
    "If the user asks to see or load all memories, use 'load_all_memory'. "
    "If the user tells you a fact about themselves or the system, use 'save_memory' to remember it. "
    "If you need to recall a known fact, use 'get_memory'. "
    "Don't just make things up. "
    "Use the available tools if they help you solve a problem. "
    "Work efficiently by chaining commands together. "
    "Always provide clear and professional answers. "
    "Load all memory entries if you're missing information. "
)

def get_dynamic_system_prompt() -> str:
    """Returns the base prompt appended with the current date and time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{BASE_SYSTEM_PROMPT}\n\n[Current Timestamp: {now}]"

TOKEN_THRESHOLD = 2000

# ANSI Farben
BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# --- HELFER ---

def get_token_count(messages: list) -> int:
    total_chars = 0
    for msg in messages:
        total_chars += len(msg["content"])
    return total_chars // 4

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_memory_to_file(memory_dict):
    with open(MEMORY_FILE, 'w') as f: json.dump(memory_dict, f, indent=4)

# --- TOOLS ---

def confirm_tool(tool_name: str, detail: str) -> bool:
    print(f"\n{YELLOW}[Sicherheit]{RESET} Das Tool {YELLOW}{tool_name}{RESET} soll ausgeführt werden mit: {YELLOW}{detail}{RESET}")
    choice = input(f"{YELLOW}(y/n)?{RESET} ").lower().strip()
    return choice == 'y'

@tool
def list_dir(path: str = ".") -> str:
    """Lists the files and directories in a given path."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory {path}: {str(e)}"

@tool
def save_memory(key: str, value: str) -> str:
    """Saves a piece of information (key-value pair) to long-term memory."""
    if confirm_tool("save_memory", f"key='{key}', value='{value}'"):
        memory = load_memory()
        memory[key] = value
        save_memory_to_file(memory)
        return f"Successfully remembered that {key} is {value}"
    return "[Skipped save_memory]"

@tool
def get_memory(key: str) -> str:
    """Retrieves a previously saved piece of information from memory using its key."""
    memory = load_memory()
    if key in memory: return f"The {key} is {memory[key]}"
    return f"I don't have any memory regarding '{key}'."

@tool
def load_all_memory() -> str:
    """Reads all entries currently stored in the long-term memory."""
    memory = load_memory()
    if not memory:
        return "The memory is currently empty."
    return str(memory)

@tool
def read_file(path: str) -> str:
    """Read the content of a file at a given path."""
    if confirm_tool("read_file", f"path='{path}'"):
        try:
            with open(path, 'r') as f: return f.read()
        except Exception as e: return f"Error reading file {path}: {str(e)}"
    return "[Skipped read_file]"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at a given path. Overwrites if exists."""
    preview = (content[:30] + '...') if len(content) > 30 else content
    if confirm_tool("write_file", f"path='{path}', content='{preview}'"):
        try:
            with open(path, 'w') as f: f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e: return f"Error writing to file {path}: {str(e)}"
    return "[Skipped write_file]"

@tool
def bash_command(command: str) -> str:
    """Execute a bash command and return the output."""
    if confirm_tool("bash_command", f"command='{command}'"):
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
        except Exception as e: return f"Exception: {str(e)}"
    return "[Skipped bash_command]"

# --- CHAT LOGIK ---

def colorize_content(text: str) -> list:
    """
    Zerlegt den Text in Fragmente und weist Farben zu.
    - Codeblocks (```python, ```bash, etc.): Cyan
    - Tabellen (| ... |): Hellgrau/Blau zur Abhebung
    - Header H1 (#): Braun
    - Header H2 (##): Purple
    - Header H3+ (###...): Orange + Zeilenumbruch
    - Quotes (> ...): Dunkelgrau als zusammenhängender Block
    - Alles andere: Standard
    """
    fragments = []
    lines = text.split('\n')

    in_code_block = False
    current_block = []
    current_color = RESET

    in_table_block = False
    table_color = "\033[37m" # Hellgrau für Tabellen

    in_quote_block = False
    quote_color = "\033[90m" # Dunkelgrau für Quotes

    def flush_block(block_lines, color):
        if not block_lines: return
        content = "\n".join(block_lines)
        fragments.append((content, color))

    for line in lines:
        stripped = line.strip()

        # --- FALL 1: CODEBLOCK-LOGIK ---
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                lang = stripped.replace("```", "").strip().lower()
                if lang in ["python", "bash", "sh", "shell", "javascript", "cpp", "sql"]:
                    current_color = CYAN
                else:
                    current_color = RESET

                first_line_content = line.replace("```", "").strip()
                if first_line_content:
                    current_block.append(first_line_content)
            else:
                last_line_content = line[:line.rfind("```")].strip()
                if last_line_content:
                    current_block.append(last_line_content)
                flush_block(current_block, current_color)
                current_block = []
                in_code_block = False
                fragments.append(("\n", RESET))
            continue

        if in_code_block:
            current_block.append(line)
            continue

        # --- FALL 2: TABELLEN-LOGIK ---
        is_table_line = stripped.startswith("|")
        if is_table_line:
            if not in_table_block:
                in_table_block = True
                current_block = [line]
            else:
                current_block.append(line)
            continue
        elif in_table_block:
            flush_block(current_block, table_color)
            current_block = []
            in_table_block = False

        # --- FALL 3: BLOCKQUOTE-LOGIK ---
        is_quote_line = stripped.startswith(">")
        if is_quote_line:
            if not in_quote_block:
                in_quote_block = True
                current_block = [line]
            else:
                current_block.append(line)
            continue
        elif in_quote_block:
            flush_block(current_block, quote_color)
            current_block = []
            in_quote_block = False

        # --- FALL 4: HEADER-LOGIK (Erweitert für H1, H2, H3+) ---
        # Wir zählen die Anzahl der '#' am Anfang
        header_match = re.match(r'^(#+)\s(.*)', line)
        if header_match:
            hashes = header_match.group(1)
            level = len(hashes)

            if level == 1:
                header_color = "\03int[38;5;94m" # Braun (Sienna/Brown)
            elif level == 2:
                header_color = "\033[38;5;135m" # Purple
            else:
                header_color = "\033[38;5;208m" # Orange für H3+

            fragments.append((line, header_color))
            fragments.append(("\n", RESET))
            continue

        # --- FALL 5: NORMALER TEXT ---
        if line:
            fragments.append((line, RESET))
        else:
            fragments.append(("\n", RESET))

    # Cleanup am Ende des Loops
    if in_code_block:
        flush_block(current_block, current_color)
    if in_table_block:
        flush_block(current_block, table_color)
    if in_quote_block:
        flush_block(current_block, quote_color)

    return fragments

def summarize_history(history: list, model: ChatOpenAI) -> list:
    """
    Implements a Summary Buffer Memory:
    Summarizes the older part of the conversation and keeps the 2 most recent messages intact.
    """
    if len(history) <= 3: # Wenn wir weniger als 3 Nachrichten haben, brauchen wir noch keine Zusammenfassung
        return history

    print(f"\n{GREEN}[System]{RESET} Komprimierung der Historie (Summary Buffer)...")

    # Wir trennen die Historie in:
    # 1. Alles außer den letzten 2 Nachrichten
    # 2. Die letzten 2 Nachrichten
    to_summarize = history[:-2]
    recent_messages = history[-2:]

    # Erstelle den Text für die Zusammenfassung aus dem alten Teil
    history_text = "".join([f"{m['role']}: {m['content']}\n" for m in to_summarize])
    summary_prompt = f"Summarize this conversation briefly:\n\n{history_text}"

    try:
        summary_res = model.invoke(summary_prompt)
        # Das neue Format ist: [System Summary, Message -2, Message -1]
        new_history = [
            {"role": "system", "content": f"Summary of previous conversation: {summary_res.content}"},
            recent_messages[0],
            recent_messages[1]
        ]
        return new_history
    except Exception as e:
        # Zeige den genauen Fehler an, damit wir wissen, ob es ein Timeout oder Connection Error ist
        print(f"  -> [Summarize Error]: {e}")
        return history

def load_history(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f: return json.load(f)
        except: return []
    return []

def save_history(history, filename):
    with open(filename, 'w') as f: json.dump(history, f, indent=4)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded = json.load(f)
                # Sicherstellen, dass alle Keys vorhanden sind (Merge mit Defaults)
                for key, value in DEFAULT_CONFIG.items():
                    if key not in loaded:
                        loaded[key] = value
                return loaded
        except: return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def chat_interface():
    global TOKEN_THRESHOLD

    # --- INITIALISIERUNG ---
    config = load_config()
    TOKEN_THRESHOLD = config["token_threshold"]
    BASE_URL = config["base_url"]
    API_KEY = config["api_key"]

    # Das Modell wird erst erstellt, wenn die URL aus der Config bekannt ist!
    local_model = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY) #, temperature=0.7)
    tools_list = [read_file, write_file, bash_command, save_memory, get_memory, load_all_memory, list_dir]
    agent = create_agent(model=local_model, tools=tools_list, system_prompt=get_dynamic_system_prompt())

    chat_history = load_history(HISTORY_FILE)

    if not any(m['role'] == 'system' for m in chat_history):
        chat_history.insert(0, {"role": "system", "content": get_dynamic_system_prompt()})

    print(f"\n{CYAN}Willkommen im Eternal Chat!{RESET}")
    print(f"Modell-URL: {YELLOW}{BASE_URL}{RESET}")
    print(f"Nutze {YELLOW}/help{RESET} für eine Liste aller Befehle.\n")

    while True:

        user_input = input(f"{BLUE}Du:{RESET}\n").strip()
        if not user_input: continue

        # --- COMMAND HANDLING ---
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/help":
                print(f"\n{MAGENTA}--- Verfügbare Befehle ---{RESET}")
                print(f"{CYAN}/exit{RESET}          - Beendet die Session und speichert.")
                print(f"{CYAN}/clear{RESET}         - Löscht die aktuelle Chat-Historie.")
                print(f"{CYAN}/show{RESET}          - Zeigt die aktuelle Historie an.")
                print(f"{CYAN}/memory{RESET}        - Zeigt alle gespeicherten Fakten an.")
                print(f"{CYAN}/shrink{RESET}        - Komprimiert die Historie manuell.")
                print(f"{CYAN}/threshold <n>{RESET} - Setzt das Token-Limit auf <n>.")
                print(f"{CYAN}/tools{RESET}         - Zeigt alle verfügbaren Tools an.")
                print(f"{CYAN}/tokens{RESET}        - Zeigt die aktuelle Tokenanzahl an.")
                print(f"{CYAN}/config{RESET}        - Zeigt die aktuelle config an.")
                print(f"{CYAN}/url{RESET}           - Setzt die Connection-URL.")
                print(f"{CYAN}/help{RESET}          - Zeigt diese Hilfe an.")
                print(f"{MAGENTA}--------------------------{RESET}\n")
                continue

            elif cmd == "/exit":
                save_history(chat_history, HISTORY_FILE)
                print("Konversation gespeichert. Tschüss!")
                break

            elif cmd == "/clear":
                chat_history = []
                save_history(chat_history, HISTORY_FILE)
                print("Historie gelöscht.\n")

            elif cmd == "/show":
                print(f"\n{MAGENTA}--- Historie ---{RESET}")
                for m in chat_history:
                    color = BLUE if m['role'] == 'user' else (RED if m['role'] == 'assistant' else GREEN)
                    print(f"{color}{m['role'].upper()}:{RESET} {m['content']}")
                print(f"{MAGENTA}----------------\n")

            elif cmd == "/memory":
                mem = load_memory()
                print(f"\n{MAGENTA}--- Memory ---{RESET}")
                if not mem: print("(Leer)")
                for k, v in mem.items(): print(f"{CYAN}{k}{RESET} = {v}")
                print(f"{MAGENTA}--------------\n")

            elif cmd == "/shrink":
                chat_history = summarize_history(chat_history, local_model)
                save_history(chat_history, HISTORY_FILE)
                print("Historie komprimiert.\n")

            elif cmd == "/threshold":
                if len(parts) > 1:
                    try:
                        TOKEN_THRESHOLD = int(parts[1])
                        print(f"Threshold auf {TOKEN_THRESHOLD} gesetzt.\n")
                    except: print("Fehler: Zahl erforderlich.")
                else: print(f"Aktuell: {TOKEN_THRESHOLD}")

            elif cmd == "/tools":
                print(f"\n{MAGENTA}--- Verfügbare Tools ---{RESET}")
                # Wir extrahieren die Namen der Tools aus der tools_list
                for t in tools_list:
                    # t ist das Tool-Objekt, wir nehmen den Namen
                    print(f"{CYAN}{t.name}{RESET}: {t.description}")
                print(f"{MAGENTA}------------------------{RESET}\n")
                continue

            elif cmd == "/tokens":
                current_tokens = get_token_count(chat_history)
                print(f"\n{CYAN}Aktuelle Tokenanzahl:{RESET} ~{current_tokens}")
                print(f"{CYAN}Limit:{RESET} {TOKEN_THRESHOLD}")
                print(f"{CYAN}Auslastung:{RESET} {(current_tokens / TOKEN_THRESHOLD) * 100:.1f}%")
                print(f"{MAGENTA}------------------{RESET}\n")
                continue

            elif cmd == "/url": # Hilfsbefehl um die URL zu ändern
                if len(parts) > 1:
                    new_url = parts[1]
                    BASE_URL = new_url
                    config["base_url"] = new_url
                    save_config(config)
                    # WICHTIG: Das Modell muss neu initialisiert werden, damit die URL greift!
                    local_model = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, temperature=0.2)
                    print(f"URL auf {BASE_URL} gesetzt. (Modell neu geladen)\n")
                else: print(f"Aktuell: {BASE_URL}")

            elif cmd == "/config":
                print(f"\n{MAGENTA}--- Aktuelle Konfiguration ---{RESET}")
                print(f"{CYAN}Token-Threshold:{RESET} {TOKEN_THRESHOLD}")
                print(f"{CYAN}Base-URL:{RESET}      {BASE_URL}")
                print(f"{CYAN}API-Key:{RESET}       {API_KEY}")
                print(f"{MAGENTA}------------------------------{RESET}\n")
                continue

            else:
                print(f"Unbekannter Befehl: {cmd}")
            continue

        # --- CHAT FLOW ---
        chat_history.append({"role": "user", "content": user_input})
        new_sys_prompt = get_dynamic_system_prompt()
        system_index = -1
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'system':
                system_index = i
                break

        if system_index != -1:
            chat_history[system_index] = {"role": "system", "content": new_sys_prompt}
        else:
            chat_history.insert(0, {"role": "system", "content": new_sys_prompt})

        if get_token_count(chat_history) > TOKEN_THRESHOLD:
            chat_history = summarize_history(chat_history, local_model)
        else:
            print(f"{YELLOW}(Tokens: ~{get_token_count(chat_history)}){RESET}")

        try:
            response = agent.invoke({"messages": chat_history})
            assistant_content = response["messages"][-1].content

            # LAYOUT-ÄNDERUNG: Zeilenumbruch nach "Assistant:" + Farbiges Rendering
            print(f"{RED}Assistant:{RESET}")

            fragments = colorize_content(assistant_content)
            for text, color in fragments:
                # Wir drucken das Fragment mit der zugeteilten Farbe
                # WICHTIG: Wir nutzen end="" damit die Zeilenumbrüche im Text erhalten bleiben
                print(f"{color}{text}{RESET}", end="")

            print("\n") # Der abschließende Zeilenumbruch nach dem Block

            chat_history.append({"role": "assistant", "content": assistant_content})
            save_history(chat_history, HISTORY_FILE)
        except Exception as e:
            print(f"\n[Fehler]: {e}")
            if chat_history[-1]["role"] != "user": chat_history.append({"role": "user", "content": user_input})

if __name__ == "__main__":
    chat_interface()
