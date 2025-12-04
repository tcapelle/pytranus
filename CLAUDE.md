# General Cape Claude Code Guidelines
This is my personal Claude Instructions, follow them when developing projects.

## Coding guidelines and philosophy
- You should generate code that is simple and redable, avoid unnecesary abstractions and complexity. This is a research codebase so we want to be mantainable and readable.
- Avoid overly defensive coding, no need for a lot of `try, except` patterns, I want the code to fail is something is wrong so that i can fix it.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.

## Dependency management
Use uv as dependency manager for python. Run scripts using `uv run script.py` instead of calling pythong directly. This is also true for tools like `uv run pytest`
 
## Argument parsing
Use `simple_parsing` as an argument paraser for the scripts. Like this
 
```python
import simple_parsing as sp

@dataclass
class Args:
    """ Help string for this group of command-line arguments """
    arg1: str       # Help string for a required str argument
    arg2: int = 1   # Help string for arg2

args = sp.parse(Args)
```
## Typing
We are using modern python (3.12+) so no need to import annotations, you can also use `dict` and `list` and `a | b` or `a | None` instead of Optional, Union, Dict, List, etc...
 
## Printing and logging
Use rich.Console to print stuff on scripts, use Panel and console.rule to make stuff organized
 
## Debugging
When running scripts, use the `debug` flags if avaliable, and ask to run the full pipeline (this enables faster iteration)
 
# Code Review

## Remove AI code slop

After finishing a code review, always check the diff against main, and remove all AI generated slop introduced in this branch.

This includes:
- Extra comments that a human wouldn't add or is inconsistent with the rest of the file
- Variables that are used a single time right after declaration, prefer inlining the rhs
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase (especially if called by trusted / validated codepaths)
- Casts to any to get around type issues
- Any other style that is inconsistent with the file

Report at the end with only a 1-3 sentence summary of what you changed